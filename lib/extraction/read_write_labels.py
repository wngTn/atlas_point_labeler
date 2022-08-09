from trimesh import PointCloud
import point_labeling.extract_person as extract_person
import point_labeling.labels as labels
import point_labeling.extract_floor as extract_floor
import point_labeling.extract_wall as extract_wall
import glob
import os
import re
import open3d as o3d
import numpy as np
from tqdm import tqdm


def read_labels(path):
    """
    Reads all the label files in the path parameter
    Returns dictionary with frame_id and its label array
    """
    label_file_names = list(sorted(glob.glob(path + '/**/labels/*.label', recursive=True)))
    id_frame_dic = {}
    for label_file_name in label_file_names:
        frame_id = int(re.search(r'\d+', os.path.basename(label_file_name))[0])
        arr = labels.read_labels(label_file_name)
        id_frame_dic[frame_id] = arr
    
    return id_frame_dic

    

def read_point_clouds(path):
    """
    Reads all the point cloud files in the path parameter
    Returns dictionary with frame_id and its point cloud
    """
    point_cloud_file_names = list(sorted(glob.glob(path + '/**/*.ply', recursive=True)))
    id_frame_dic = {}
    for point_cloud_file_name in point_cloud_file_names:
        frame_id = int(re.search(r'\d+', os.path.basename(point_cloud_file_name))[0])
        pcd = o3d.io.read_point_cloud(point_cloud_file_name)
        id_frame_dic[frame_id] = pcd
    
    return id_frame_dic



def prune_labels(labels, point_cloud, config, z_floor_coordinate, min_x, min_y, max_x, max_y):
    assert(len(labels) == len(point_cloud.points))

    labels = [0] * len(labels)

    # mesh_kp = extract_points.prune_point_clouds(config)
    # pruned_points_ind = []
    # for _, (_, pcd_ind) in mesh_kp.items():
    #     pruned_points_ind.append(pcd_ind[:, 3].flatten())

    # pruned_points_ind = np.concatenate(pruned_points_ind, axis=0)

    # print("Iterating through every point cloud point...")

    # print(f"There are {len(pruned_points_ind)} many indices")
    # for index in pruned_points_ind:
    #     labels[int(index)] = 3

    floor_indices = extract_floor.get_floor_indices(point_cloud, z_floor_coordinate)
    for floor_index in floor_indices:
        labels[floor_index] = 2

    wall_indices = extract_wall.get_wall_indices(point_cloud, min_x, min_y, max_x, max_y)
    for wall_index in wall_indices:
        labels[wall_index] = 1
    

    return labels



def r_w_labels(path, config):
    label_dic = read_labels(path)
    point_cloud_dic = read_point_clouds(path)

    assert(len(label_dic) == len(point_cloud_dic))

    
    first_pcd = list(point_cloud_dic.values())[0]


    floor_z_coordinate = extract_floor.get_floor(first_pcd)
    min_x, min_y, max_x, max_y = extract_wall.get_wall(first_pcd, nb_points=350, radius=0.1)


    for k, _ in label_dic.items():
        labels_from_dic = label_dic[k]
        point_cloud_from_dic = point_cloud_dic[k]
        config.DEBUG.FRAME_ID = k
        pruned_label = prune_labels(labels_from_dic, point_cloud_from_dic, config, floor_z_coordinate, min_x, min_y, max_x, max_y)
        labels.write_labels(os.path.join(path, "pruned_labels", f"{k}_pointcloud.label"), pruned_label)
        print(f"Written label for frame number {k}")



def debug(path, config):

    f_id = 5

    label_dic = read_labels(path)
    point_cloud_dic = read_point_clouds(path)

    floor = extract_floor.get_floor(point_cloud_dic[f_id])
    config.DEBUG.FRAME_ID = f_id

    label = label_dic[f_id]
    pcd = point_cloud_dic[f_id]

    min_x, min_y, max_x, max_y = extract_wall.get_wall(pcd, nb_points=150, radius=0.1)

    # min_x, min_y, max_x, max_y = extract_wall.get_wall(pcd)

    indices = extract_wall.get_wall_indices(pcd, min_x, min_y, max_x, max_y)
    floor_indices = extract_floor.get_floor_indices(pcd, floor)

    for index in indices:
        pcd.colors[index] = [1, 0, 0]
    
    for floor_index in floor_indices:
        pcd.colors[floor_index] = [0, 1, 0]


    



    # p_label = prune_labels(label_dic[f_id], point_cloud_dic[f_id], config, floor, min_x, min_y, max_x, max_y)


    # for i, label in enumerate(p_label):
    #     if label == 3:
    #         pcd.colors[i] == [0, 0, 1]
    #     if label == 2:
    #         pcd.colors[i] = [0, 1, 0]
    #     if label == 1:
    #         pcd.colors[i] = [1, 0, 0]


    
    # p = np.array(p)
    # print(f"There are {len(p)} many points in the o3d visualizer")

    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(p)
    # pcd2.colors = o3d.utility.Vector3dVector(c)
    o3d.visualization.draw_geometries([pcd])


def remove_clouds_outliers(pcd):
    cl, ind = pcd.remove_radius_outlier(nb_points=150, radius=0.1)
    pcd = pcd.select_by_index(ind)

    return pcd



    





