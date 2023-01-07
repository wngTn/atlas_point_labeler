import open3d as o3d
import numpy as np
import copy
import open3d.visualization.gui as gui
from scipy.spatial.transform import Rotation as rot
from tqdm import tqdm
import os
from scipy.spatial.transform import Rotation as rot
import extraction.labels as labels
import json

import logging

logger = logging.getLogger(__name__)


def remove_noise(trial, phase, anno_frame_ids, point_cloud_dir, labels_dir, cal_id):
    indices_path = os.path.join(point_cloud_dir, "..", "indices")
    os.makedirs(indices_path, exist_ok=True)
    cal_file = os.path.join('data', "calibrations", f'calibration_{cal_id}.json')
    if not os.path.exists(cal_file):
        logger.warn(
            f"There was no calibration file for {trial}_{phase} in {cal_file}. Skipping wall extraction..."
        )
        return

    with open(cal_file, 'r') as f:
        calibration_dict = json.load(f)

    for frame_id in tqdm(anno_frame_ids, desc="Removing Noise..."):
        if not os.path.exists(
                os.path.join(point_cloud_dir, f"{str(frame_id).zfill(4)}_pointcloud.ply")):
            continue

        pcd_filename = os.path.join(point_cloud_dir, f"{str(frame_id).zfill(4)}_pointcloud.ply")
        label_filename = os.path.join(labels_dir, f"{str(frame_id).zfill(4)}_pointcloud.label")
        point_cloud = o3d.io.read_point_cloud(pcd_filename)

        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

        with open(os.path.join(indices_path, f"{str(frame_id).zfill(4)}.npy"), 'wb') as f:
            np.save(f, np.array(ind).astype('uint32'), allow_pickle=True)
            print("Saved", indices_path, f"{str(frame_id).zfill(4)}.npy")
        
        inlier_cloud = point_cloud.select_by_index(ind)
        o3d.io.write_point_point_cloud(pcd_filename, inlier_cloud)
        
        labels.delete_labels(label_filename, np.array(ind))