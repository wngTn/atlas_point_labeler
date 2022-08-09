import open3d as o3d
import numpy as np
import copy
import open3d.visualization.gui as gui
from tqdm import tqdm
import os
import extraction.labels as labels
import json

import logging
logger = logging.getLogger(__name__)


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * .5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def get_floor(point_cloud):

    voxel_size = 0.05

    points = np.asarray(point_cloud.points)
    point_cloud = point_cloud.select_by_index(np.where(points[:, 2] < 1)[0])

    # create 4m * 4m triangle mesh
    vertices = np.array([
        [1.5, 1.5, 0],
        [1.5, -1.5, 0],
        [-1.5, -1.5, 0],
        [-1.5, 1.5, 0]
    ])

    faces = np.array([
        [0, 1, 2],
        [2, 3, 0]
    ])

    floor_mesh = o3d.geometry.TriangleMesh()
    floor_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    floor_mesh.triangles = o3d.utility.Vector3iVector(faces)

    floor_pcd = floor_mesh.sample_points_uniformly(number_of_points=10000)
    floor_pcd = floor_mesh.sample_points_poisson_disk(number_of_points=3000, pcl=floor_pcd)

    source_down, source_fpfh = preprocess_point_cloud(floor_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(point_cloud, voxel_size)


    result = execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size)

    floor_pcd.transform(result.transformation)

    # o3d.visualization.draw_geometries([floor_pcd, point_cloud])


    floor_z_coordinate = np.average(np.array(floor_pcd.points), axis=0)[2]

    return floor_z_coordinate


def get_floor_indices(point_cloud, z_floor_coordinate):
    buffer_min = 0.05
    buffer_max = 0.05

    point_cloud_copy = copy.deepcopy(point_cloud)

    only_indices = np.arange(len(point_cloud.points))
    # [ [x, y, z, index], [x, y, z, index], ...]
    #
    #

    # cropping
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(mins[0], mins[1], mins[2]), max_bound=(maxs[0], maxs[1], maxs[2]))
    # cropped_pcd = point_cloud_copy.crop(bbox)
    max_mask = np.asarray(point_cloud_copy.points)[:, 2] < (z_floor_coordinate + buffer_max)
    min_mask = np.asarray(point_cloud_copy.points)[:, 2] > (z_floor_coordinate - buffer_min)

    mask = np.logical_and(max_mask, min_mask)

    only_indices = only_indices[mask]

    # pcd = point_cloud_copy.select_by_index(only_indices)
    # o3d.visualization.draw_geometries([pcd])

    return only_indices


def extract_floor(trial, anno_frame_ids, point_cloud_dir, labels_dir):
    cal_file = os.path.join('data', 'label_data', trial, 'calibrations.json')
    if not os.path.exists(cal_file):
        logger.warn(f"There was no calibration file for {trial} in {cal_file}. Skipping wall extraction...")
        return

    with open(cal_file, 'r') as f:
        calibration_dict = json.load(f)

    for frame_id in tqdm(anno_frame_ids, desc="Extracting the floor..."):
        pcd_filename = os.path.join(point_cloud_dir, f"{str(frame_id).zfill(4)}_pointcloud.ply")
        label_filename = os.path.join(labels_dir, f"{str(frame_id).zfill(4)}_pointcloud.label")
        point_cloud = o3d.io.read_point_cloud(pcd_filename)

        indices =  get_floor_indices(point_cloud, calibration_dict["floor"])
        labels.overwrite_labels(label_filename, indices, 1)
