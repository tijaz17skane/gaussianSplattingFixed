import os
import json
import math
import argparse
import csv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import datetime

# --- Constants ---
PLANAR_WIDTH = 2454
PLANAR_HEIGHT = 1840

# --- Helper Functions ---
def equirectangular_to_perspective(img, fov_deg, yaw_deg, pitch_deg, width, height):
    """
    Projects an equirectangular image to a perspective (rectilinear) image.
    yaw_deg: center direction (degrees, 0 = forward)
    pitch_deg: up/down (degrees, 0 = horizontal)
    """
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Focal length in pixels
    fx = fy = width / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2

    # Build direction vectors for each pixel
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)
    x_cam = (xv - cx) / fx
    y_cam = (yv - cy) / fy
    z_cam = np.ones_like(x_cam)
    dirs = np.stack([x_cam, -y_cam, z_cam], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Rotate by pitch then yaw
    rot = R.from_euler('yx', [yaw_deg, pitch_deg], degrees=True)
    dirs_rot = rot.apply(dirs.reshape(-1, 3)).reshape(height, width, 3)

    # Convert to spherical coordinates
    lon = np.arctan2(dirs_rot[..., 0], dirs_rot[..., 2])
    lat = np.arcsin(dirs_rot[..., 1])

    # Map to equirectangular pixel coordinates
    equ_h, equ_w = img.shape[:2]
    uf = (lon / np.pi + 1) / 2 * equ_w
    vf = (0.5 - lat / np.pi) * equ_h

    map_x = uf.astype(np.float32)
    map_y = vf.astype(np.float32)
    persp = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp

def compute_intrinsics(width, height, fov_deg):
    fov = math.radians(fov_deg)
    fx = fy = width / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

def quaternion_from_yaw(yaw_deg):
    # Returns quaternion (x, y, z, w) for rotation about Y axis (yaw)
    rot = R.from_euler('y', yaw_deg, degrees=True)
    q = rot.as_quat()  # x, y, z, w
    return q.tolist()

def rpy_from_quaternion(q):
    rot = R.from_quat(q)
    return rot.as_euler('xyz', degrees=True).tolist()

def main():
    parser = argparse.ArgumentParser(description='Convert spherical images to planar perspective images at multiple yaw angles.')
    parser.add_argument('--spherical_dir', default='/mnt/data/tijaz/dataSets/data/section_3/spherical_images', help='Input folder containing spherical images (default: spherical_images)')
    parser.add_argument('--meta_json', default='/mnt/data/tijaz/dataSets/data/section_3/meta.json', help='Input meta.json file (default: meta.json)')
    parser.add_argument('--output_dir', default='/mnt/data/tijaz/dataSets/data/section_3/images', help='Output folder for planar images (default: converted_images)')
    parser.add_argument('--output_meta_json', default='/mnt/data/tijaz/dataSets/data/section_3/meta.json', help='Output meta.json file (default: converted.json)')
    parser.add_argument('--calibration_csv', default='/mnt/data/tijaz/dataSets/data/section_3/calibration.csv', help='Output calibration CSV file (default: calibration2.csv)')
    parser.add_argument('--fov', type=float, default=60, help='Field of view in degrees (default: 60)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load meta.json
    with open(args.meta_json, 'r') as f:
        meta = json.load(f)

    # Determine where the spherical images are stored
    if isinstance(meta, dict) and 'spherical_images' in meta:
        meta_list = meta['spherical_images']
        meta_is_dict = True
    elif isinstance(meta, list):
        meta_list = meta
        meta_is_dict = False
    else:
        print("Error: meta.json structure not recognized. Exiting.")
        return

    # List all images in the spherical_images folder
    sph_img_files = [f for f in os.listdir(args.spherical_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sph_img_files.sort()

    # Build a lookup for extrinsics in meta.json
    extrinsics_lookup = {}
    for entry in meta_list:
        # Match by filename (basename of path)
        entry_basename = os.path.basename(entry['path'])
        extrinsics_lookup[entry_basename] = entry

    today_str = datetime.datetime.now().isoformat()

    # Prepare calibration CSV
    calib_rows = []
    new_meta_entries = []
    yaw_intrinsics = {}  # yaw_label -> (fx, fy, cx, cy)

    total_imgs = len(sph_img_files)
    n_orient = int(math.ceil(360.0 / args.fov))
    for i in range(n_orient):
        yaw = i * args.fov
        yaw_label = f"yaw{int(yaw):03d}"
        fx, fy, cx, cy = compute_intrinsics(PLANAR_WIDTH, PLANAR_HEIGHT, args.fov)
        yaw_intrinsics[yaw_label] = (fx, fy, cx, cy)

    for idx, img_file in enumerate(sph_img_files):
        print(f"Processing {idx+1} of {total_imgs}: {img_file}")
        if img_file not in extrinsics_lookup:
            print(f"  Warning: No extrinsics found in meta.json for {img_file}, skipping.")
            continue
        entry = extrinsics_lookup[img_file]
        img_path = os.path.join(args.spherical_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            continue
        base_id = os.path.splitext(img_file)[0].replace('ladybug_', 'lb_')
        # Determine sensor type
        img_file_lower = img_file.lower()
        if 'front' in img_file_lower:
            sensor_type = 'ladybug_front'
        elif 'back' in img_file_lower:
            sensor_type = 'ladybug_back'
        else:
            sensor_type = 'ladybug'

        for i in range(n_orient):
            yaw = i * args.fov
            yaw_label = f"yaw{int(yaw):03d}"
            fov_label = f"fov{int(args.fov)}"
            out_name = f"{yaw_label}_{fov_label}_{base_id}.jpg"
            out_path = os.path.join(args.output_dir, out_name)
            # Project
            persp = equirectangular_to_perspective(img, args.fov, yaw, 0, PLANAR_WIDTH, PLANAR_HEIGHT)
            cv2.imwrite(out_path, persp)
            # New meta entry
            q = quaternion_from_yaw(yaw)
            rpy = rpy_from_quaternion(q)
            sensor_id = f"{yaw_label}_{fov_label}_{sensor_type}"
            new_entry = {
                'sensor_id': sensor_id,
                'path': out_name,
                'fov': args.fov,
                'time_stamp': entry['time_stamp'],
                'pose': {
                    'translation': entry['pose']['translation'],
                    'orientation_xyzw': q,
                    'orientation_roll_pitch_yaw': rpy
                }
            }
            new_meta_entries.append(new_entry)

    # Write calibration CSV (append to existing file if present)
    calib_fieldnames = [
        'id','sensor_id','ipm_ignore','intr_calibration_date',
        'focal_length_px_x','focal_length_px_y','principal_point_px_x','principal_point_px_y',
        'lens_distortion_calibration_date','calibration_type',
        'k1','k2','k3','p1','p2','s1','s2'
    ]
    existing_sensor_ids = set()
    next_id = 0
    if os.path.exists(args.calibration_csv):
        with open(args.calibration_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_sensor_ids.add(row['sensor_id'])
                try:
                    next_id = max(next_id, int(row['id']) + 1)
                except Exception:
                    pass
    with open(args.calibration_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=calib_fieldnames)
        # If file is empty, write header
        if os.stat(args.calibration_csv).st_size == 0:
            writer.writeheader()
        for yaw_label, (fx, fy, cx, cy) in yaw_intrinsics.items():
            for sensor_type in ['ladybug_front', 'ladybug_back', 'ladybug']:
                fov_label = f"fov{int(args.fov)}"
                sensor_id = f'{yaw_label}_{fov_label}_{sensor_type}'
                if sensor_id in existing_sensor_ids:
                    continue
                writer.writerow({
                    'id': next_id,
                    'sensor_id': sensor_id,
                    'ipm_ignore': 'False',
                    'intr_calibration_date': today_str,
                    'focal_length_px_x': fx,
                    'focal_length_px_y': fy,
                    'principal_point_px_x': cx,
                    'principal_point_px_y': cy,
                    'lens_distortion_calibration_date': today_str,
                    'calibration_type': 'opencv',
                    'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0, 's1': 0.0, 's2': 0.0
                })
                next_id += 1

    # Update meta.json
    if meta_is_dict:
        # Only append if meta['spherical_images'] exists and is a list
        if isinstance(meta, dict) and 'spherical_images' in meta and isinstance(meta['spherical_images'], list):
            meta['spherical_images'].extend(new_meta_entries)
        else:
            print("Error: meta['spherical_images'] is not a list or missing. Cannot append extrinsics.")
    elif isinstance(meta, list):
        meta.extend(new_meta_entries)
    else:
        print("Error: meta.json structure not recognized for appending extrinsics.")
    with open(args.output_meta_json, 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == '__main__':
    main() 