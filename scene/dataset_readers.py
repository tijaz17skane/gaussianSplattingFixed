#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import laspy
from scipy.spatial.transform import Rotation as R

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    depth_params: Optional[dict]
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: Optional[object]
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def quaternion_to_rotation_matrix(q_xyzw):
    r = R.from_quat(q_xyzw)
    return r.as_matrix()

def read_las_pointcloud(las_path):
    las = laspy.read(las_path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    if hasattr(las, 'red'):
        rgb = np.vstack([las.red, las.green, las.blue]).T
        if rgb.max() > 255:
            rgb = (rgb / 65535.0 * 255).astype(np.uint8)
    else:
        rgb = np.zeros_like(xyz)
    return xyz, rgb

def readCustomMetaSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    import json
    import os
    import laspy
    import csv
    from scipy.spatial.transform import Rotation as R
    from utils.graphics_utils import focal2fov
    # 1. Parse meta.json
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    # 2. Parse calibration.csv
    calib_path = os.path.join(path, "calibration.csv")
    sensor_intrinsics = {}
    with open(calib_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sensor_id = row['sensor_id']
            sensor_intrinsics[sensor_id] = {
                'width': None,  # will be filled per image
                'height': None, # will be filled per image
                'fx': float(row['focal_length_px_x']),
                'fy': float(row['focal_length_px_y']),
                'cx': float(row['principal_point_px_x']),
                'cy': float(row['principal_point_px_y']),
                'k1': float(row['k1']),
                'k2': float(row['k2']),
                'k3': float(row['k3']),
                'p1': float(row['p1']),
                'p2': float(row['p2']),
                's1': float(row['s1']),
                's2': float(row['s2']),
            }
    # 3. Read point cloud (auto-detect .las or .laz)
    las_candidates = [f for f in os.listdir(path) if f.lower().endswith('.las') or f.lower().endswith('.laz')]
    if not las_candidates:
        raise FileNotFoundError("No .las or .laz file found in dataset root.")
    las_path = os.path.join(path, las_candidates[0])
    las = laspy.read(las_path)  # laspy supports both .las and .laz
    xyz = np.vstack([las.x, las.y, las.z]).T
    if hasattr(las, 'red'):
        rgb = np.vstack([las.red, las.green, las.blue]).T
        if rgb.max() > 255:
            rgb = (rgb / 65535.0 * 255).astype(np.uint8)
    else:
        rgb = np.zeros_like(xyz)
    ply_path = os.path.join(path, "pointcloud.ply")
    storePly(ply_path, xyz, rgb)

    # Print point cloud stats
    print("[DEBUG] Point cloud stats:")
    print(f"  Mean: {np.mean(xyz, axis=0)}")
    print(f"  Min: {np.min(xyz, axis=0)}")
    print(f"  Max: {np.max(xyz, axis=0)}")

    def quaternion_to_rotation_matrix(q_xyzw):
        r = R.from_quat(q_xyzw)
        return r.as_matrix()

    # 4. Scan images folder for actual image files
    images_folder = os.path.join(path, "images")
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    # Get list of actual image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    actual_images = set()
    for filename in os.listdir(images_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            actual_images.add(filename)
    
    print(f"Found {len(actual_images)} actual image files in {images_folder}")
    
    # 5. Create mapping from image paths to meta.json entries (support both 'images' and 'converted_images')
    meta_image_map = {}
    all_meta_images = meta.get("images", [])
    if "converted_images" in meta:
        all_meta_images += meta["converted_images"]
    for img in all_meta_images:
        image_filename = os.path.basename(img["path"])
        meta_image_map[image_filename] = img

    # Print total unique image filenames in meta['images'] + meta['converted_images']
    all_meta_filenames = set(os.path.basename(img['path']) for img in all_meta_images)
    print(f"[INFO] Total unique image filenames in meta['images'] + meta['converted_images']: {len(all_meta_filenames)}")

    # 6. Build CameraInfo list for all unique images in meta (if file exists)
    cam_infos = []
    processed_count = 0
    skipped_count = 0
    cam_centers_list = []
    for image_filename in all_meta_filenames:
        image_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} listed in meta.json but not found in images folder, skipping")
            skipped_count += 1
            continue
        img = meta_image_map[image_filename]
        sensor_id = str(img["sensor_id"])
        if sensor_id not in sensor_intrinsics:
            print(f"Warning: No calibration data for sensor_id '{sensor_id}', skipping {image_filename}")
            skipped_count += 1
            continue
        intr = sensor_intrinsics[sensor_id]
        q = img["pose"]["orientation_xyzw"]
        t = img["pose"]["translation"]
        Rmat = quaternion_to_rotation_matrix(q)
        Tvec = np.array(t)
        cam_centers_list.append(Tvec)
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            skipped_count += 1
            continue
        fx = intr['fx']
        fy = intr['fy']
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        cam_infos.append(CameraInfo(
            uid=img["sensor_id"],
            R=Rmat,
            T=Tvec,
            FovY=FovY,
            FovX=FovX,
            depth_params={},  # Use empty dict instead of None
            image_path=image_path,
            image_name=image_filename,
            depth_path="",
            width=width,
            height=height,
            is_test=False
        ))
        processed_count += 1
    print(f"Processed {processed_count} images, skipped {skipped_count} images")
    if processed_count == 0:
        raise ValueError("No valid images found! Check that images exist and match meta.json entries.")

    # Print camera centers stats
    if len(cam_infos) > 0:
        cam_centers = np.array(cam_centers_list)
        print("[DEBUG] Camera centers stats:")
        print(f"  Mean: {np.mean(cam_centers, axis=0)}")
        print(f"  Min: {np.min(cam_centers, axis=0)}")
        print(f"  Max: {np.max(cam_centers, axis=0)}")
    else:
        print("[DEBUG] No camera centers to print stats for.")

    # --- MINIMUM-BASED CENTERING ---
    # Compute min from point cloud
    min_xyz = np.min(xyz, axis=0)
    print(f"[DEBUG] Shifting scene by min: {min_xyz}")

    # Shift point cloud
    xyz_shifted = xyz - min_xyz
    # Shift camera centers and update CameraInfo
    cam_infos_shifted = []
    for cam in cam_infos:
        Tvec_shifted = cam.T - min_xyz
        cam_infos_shifted.append(CameraInfo(
            uid=cam.uid,
            R=cam.R,
            T=Tvec_shifted,
            FovY=cam.FovY,
            FovX=cam.FovX,
            depth_params=cam.depth_params,
            image_path=cam.image_path,
            image_name=cam.image_name,
            depth_path=cam.depth_path,
            width=cam.width,
            height=cam.height,
            is_test=cam.is_test
        ))

    # Print shifted point cloud stats
    print("[DEBUG] Shifted point cloud stats:")
    print(f"  Mean: {np.mean(xyz_shifted, axis=0)}")
    print(f"  Min: {np.min(xyz_shifted, axis=0)}")
    print(f"  Max: {np.max(xyz_shifted, axis=0)}")
    # Print shifted camera centers stats
    if len(cam_infos_shifted) > 0:
        cam_centers_shifted = np.array([c.T for c in cam_infos_shifted])
        print("[DEBUG] Shifted camera centers stats:")
        print(f"  Mean: {np.mean(cam_centers_shifted, axis=0)}")
        print(f"  Min: {np.min(cam_centers_shifted, axis=0)}")
        print(f"  Max: {np.max(cam_centers_shifted, axis=0)}")
        # Compute robust radius as max distance from origin
        radius = np.max(np.linalg.norm(cam_centers_shifted, axis=1))
        print(f"[DEBUG] Computed robust radius: {radius}")
        nerf_normalization = {"radius": float(radius)}
    else:
        print("[DEBUG] No shifted camera centers to print stats for.")
        nerf_normalization = {"radius": 1.0}

    # Save shifted point cloud
    storePly(ply_path, xyz_shifted, rgb)

    pcd = fetchPly(ply_path)
    if pcd is None:
        pcd = BasicPointCloud(points=np.zeros((0, 3)), colors=np.zeros((0, 3)), normals=np.zeros((0, 3)))
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=cam_infos_shifted,
        test_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    # Count how many images are found in 'images' and 'converted_images' in meta.json
    images_list = meta.get('images', [])
    converted_images_list = meta.get('converted_images', [])
    images_set = set(os.path.basename(img['path']) for img in images_list)
    converted_images_set = set(os.path.basename(img['path']) for img in converted_images_list)
    found_in_images = 0
    found_in_converted_images = 0
    found_in_both = 0
    for cam in cam_infos_shifted:
        fname = cam.image_name
        in_images = fname in images_set
        in_converted = fname in converted_images_set
        if in_images:
            found_in_images += 1
        if in_converted:
            found_in_converted_images += 1
        if in_images and in_converted:
            found_in_both += 1
    print(f"[INFO] Number of images successfully loaded for training: {len(cam_infos_shifted)}")
    print(f"[INFO] Of these, found in meta['images']: {found_in_images}")
    print(f"[INFO] Of these, found in meta['converted_images']: {found_in_converted_images}")
    print(f"[INFO] Of these, found in BOTH: {found_in_both}")
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "CustomMeta": readCustomMetaSceneInfo,
}