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


def prune_point_clouds(point_cloud, rotation_around_z, min_x, min_y, max_x, max_y, buffer):
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rot.from_euler('z', rotation_around_z, degrees=True).as_matrix()

    rotated_point_cloud = point_cloud.transform(rot_matrix)

    only_indices = np.arange(len(rotated_point_cloud.points))

    max_x_mask = np.logical_and(np.asarray(rotated_point_cloud.points)[:, 0] < (max_x + buffer), np.asarray(rotated_point_cloud.points)[:, 0] > (max_x - buffer))

    min_x_mask = np.logical_and(np.asarray(rotated_point_cloud.points)[:, 0] < (min_x + buffer), np.asarray(rotated_point_cloud.points)[:, 0] > (min_x - buffer))


    max_y_mask = np.logical_and(np.asarray(rotated_point_cloud.points)[:, 1] < (max_y + buffer), np.asarray(rotated_point_cloud.points)[:, 1] > (max_y - buffer))

    min_y_mask = np.logical_and(np.asarray(rotated_point_cloud.points)[:, 1] < (min_y + buffer), np.asarray(rotated_point_cloud.points)[:, 1] > (min_y - buffer))

    mask = np.logical_or(np.logical_or(max_x_mask, min_x_mask), np.logical_or(max_y_mask, min_y_mask))

    only_indices = only_indices[mask]

    return only_indices



def extract_wall(trial, anno_frame_ids, point_cloud_dir, labels_dir):
    cal_file = os.path.join('data', 'label_data', trial, 'calibrations.json')
    if not os.path.exists(cal_file):
        logger.warn(f"There was no calibration file for {trial} in {cal_file}. Skipping wall extraction...")
        return


    with open(cal_file, 'r') as f:
        calibration_dict = json.load(f)

    for frame_id in tqdm(anno_frame_ids, desc="Extracting walls..."):

        pcd_filename = os.path.join(point_cloud_dir, f"{str(frame_id).zfill(4)}_pointcloud.ply")
        label_filename = os.path.join(labels_dir, f"{str(frame_id).zfill(4)}_pointcloud.label")
        point_coud = o3d.io.read_point_cloud(pcd_filename)

        indices = prune_point_clouds(point_coud, 
            calibration_dict["rot_around_z"], 
            calibration_dict["min_x"], 
            calibration_dict["min_y"], 
            calibration_dict["max_x"], 
            calibration_dict["max_y"], 
            calibration_dict["buffer"])
        labels.overwrite_labels(label_filename, indices, 2)