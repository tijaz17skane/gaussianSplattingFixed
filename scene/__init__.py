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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], init_ply=None, init_las=None, init_laz=None, init_colmap=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # Priority: ply > las > laz > colmap > default
        if init_ply is not None:
            from plyfile import PlyData
            from scene.gaussian_model import BasicPointCloud
            print(f"Initializing Gaussians from provided .ply: {init_ply}")
            plydata = PlyData.read(init_ply)
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
            # Try to get color if present, else use zeros
            if 'red' in plydata.elements[0].data.dtype.names:
                rgb = np.stack((np.asarray(plydata.elements[0]["red"]),
                                np.asarray(plydata.elements[0]["green"]),
                                np.asarray(plydata.elements[0]["blue"])), axis=1)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
            else:
                rgb = np.zeros_like(xyz)
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))
            # Dummy cameras and normalization
            scene_info = type('SceneInfo', (), {})()
            scene_info.point_cloud = pcd
            scene_info.train_cameras = []
            scene_info.test_cameras = []
            scene_info.nerf_normalization = {"translate": np.zeros(3), "radius": 1.0}
            scene_info.ply_path = init_ply
            scene_info.is_nerf_synthetic = False
        elif init_las is not None:
            import laspy
            from scene.gaussian_model import BasicPointCloud
            print(f"Initializing Gaussians from provided .las: {init_las}")
            las = laspy.read(init_las)
            xyz = np.vstack([las.x, las.y, las.z]).T
            if hasattr(las, 'red'):
                rgb = np.vstack([las.red, las.green, las.blue]).T
                if rgb.max() > 255:
                    rgb = (rgb / 65535.0 * 255).astype(np.uint8)
                rgb = rgb / 255.0
            else:
                rgb = np.zeros_like(xyz)
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))
            scene_info = type('SceneInfo', (), {})()
            scene_info.point_cloud = pcd
            scene_info.train_cameras = []
            scene_info.test_cameras = []
            scene_info.nerf_normalization = {"translate": np.zeros(3), "radius": 1.0}
            scene_info.ply_path = init_las
            scene_info.is_nerf_synthetic = False
        elif init_laz is not None:
            import laspy
            from scene.gaussian_model import BasicPointCloud
            print(f"Initializing Gaussians from provided .laz: {init_laz}")
            las = laspy.read(init_laz)
            xyz = np.vstack([las.x, las.y, las.z]).T
            if hasattr(las, 'red'):
                rgb = np.vstack([las.red, las.green, las.blue]).T
                if rgb.max() > 255:
                    rgb = (rgb / 65535.0 * 255).astype(np.uint8)
                rgb = rgb / 255.0
            else:
                rgb = np.zeros_like(xyz)
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))
            scene_info = type('SceneInfo', (), {})()
            scene_info.point_cloud = pcd
            scene_info.train_cameras = []
            scene_info.test_cameras = []
            scene_info.nerf_normalization = {"translate": np.zeros(3), "radius": 1.0}
            scene_info.ply_path = init_laz
            scene_info.is_nerf_synthetic = False
        elif init_colmap:
            print("Initializing Gaussians from COLMAP output")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "meta.json")):
            print("Found meta.json file, assuming CustomMeta data set!")
            scene_info = sceneLoadTypeCallbacks["CustomMeta"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and not (init_ply or init_las or init_laz or init_colmap):
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.nerf_normalization = scene_info.nerf_normalization

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # Save normalized
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_normalized.ply"))
        # Save unnormalized
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), unnormalize=self.nerf_normalization)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
