import open3d as o3d
import numpy as np
from smplmodel.body_param import load_model
import copy
import open3d.visualization.gui as gui
from tqdm import tqdm
import os
import extraction.labels as labels
import json
from multiprocessing import get_context
from itertools import repeat
from itertools import chain

import logging
logger = logging.getLogger(__name__)

N_PROCESSES = os.cpu_count()


def crop_point_cloud(mesh_points, point_cloud):
    buffer_min = 0.1
    buffer_max = 0.1

    # [ [x, y, z, index], [x, y, z, index], ...]
    point_cloud_index = np.concatenate((np.asarray(point_cloud.points), np.arange(len(point_cloud.points)).reshape(-1, 1)), axis=1)

    mins = np.empty((3,))
    maxs = np.empty((3,))
    for i in range(3):
        mins[i] = np.amin(mesh_points[:, i])
        maxs[i] = np.amax(mesh_points[:, i])

    for i in range(3):
        mins[i] -= buffer_min
        maxs[i] += buffer_max

    # cropping
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(mins[0], mins[1], mins[2]), max_bound=(maxs[0], maxs[1], maxs[2]))
    # cropped_pcd = point_cloud_copy.crop(bbox)
    max_mask = (point_cloud_index[:, :3] < np.array([maxs[0], maxs[1], maxs[2]])).all(1)
    min_mask = (point_cloud_index[:, :3] > np.array([mins[0], mins[1], mins[2]])).all(1)

    mask = np.logical_and(max_mask, min_mask)

    point_cloud_index = point_cloud_index[mask]

    return point_cloud_index


def assign_pcds(meshes, point_cloud):
    """
    Creates a dictionary that couples every mesh with its cropped point clouds

    :param meshes: The meshes of the scene
    :param point_cloud: The point cloud of the scene
    :return: A list with:
        {
            Mesh_Points:np.ndarray : Point_Cloud_with_Indices:np.ndarray
        }
    """
    list_mesh_points = []
    for mesh in meshes:
        mesh_points = np.array(mesh.vertices)

        tmp_point_cloud = o3d.geometry.PointCloud()
        tmp_point_cloud.points = o3d.utility.Vector3dVector(mesh_points)
        tmp_point_cloud = tmp_point_cloud.voxel_down_sample(0.075)

        mesh_points = np.array(tmp_point_cloud.points)

        list_mesh_points.append(mesh_points)

    mesh_pcd_crop = []
    for i in range(len(list_mesh_points)):
        crop_pcd_index = crop_point_cloud(list_mesh_points[i], point_cloud)

        # add to the dictionary
        # dict: mesh_pcd : (point cloud, point cloud points index)
        mesh_pcd_crop.append((list_mesh_points[i],crop_pcd_index))

    return mesh_pcd_crop


def is_in_range(point, mesh_points, distance):
    """
    Check whether the point is in distance of any points of the mesh
    """

    # crop the mesh_points according to the buffer
    # buffer = distance
    # cropped_mesh_points = copy.deepcopy(mesh_points)
    # cropped_mesh_points[
    #     (cropped_mesh_points[:, 0] <= point[0] + buffer ) & 
    #     (cropped_mesh_points[:, 0] >= point[0] - buffer) & 
    #     (cropped_mesh_points[:, 1] <= point[1] + buffer) & 
    #     (cropped_mesh_points[:, 1] >= point[1] - buffer) & 
    #     (cropped_mesh_points[:, 2] <= point[2] + buffer) & 
    #     (cropped_mesh_points[:, 2] >= point[2] - buffer)
    # ]

    for mesh_point in mesh_points:
        if np.linalg.norm(point-mesh_point) < distance:
            return True
    return False


def indices_in_range(chunk, mesh_points, distance):
    """
    Returns the indices of the points that are in range of any of the mesh_points

    :param chunk: Array of the points coupled with the indices [[x, y, z, index], [x, y, z, index]] (<num_points>, 4)
    :param mesh_points: Array of the points of the meh
    :param distance: The maximum distance
    """
    indices = []
    for point in chunk:
        if is_in_range(point[:3], mesh_points, distance):
            indices.append(int(point[3]))

    return indices


def gen_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def prune_point_clouds(point_cloud, meshes, max_dist_to_mesh):
    # @WARNING this takes long
    mesh_pcd_crop = assign_pcds(meshes, point_cloud)
    indices = []
    for mesh_points, cropped_pcd_index in mesh_pcd_crop:
        n = len(cropped_pcd_index) // (N_PROCESSES - 1)
        chunks = gen_chunks(cropped_pcd_index, n)
        with get_context("spawn").Pool(processes=N_PROCESSES) as p:
            results = p.starmap(indices_in_range, zip(chunks, repeat(mesh_points, n), repeat(max_dist_to_mesh, n)))
        results = list(chain.from_iterable(results))
        indices.append(results)
    
    indices = list(chain.from_iterable(indices))

    return indices


# *** Some Utils Function ***

# reads a json file
def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


# reads a smpl file
def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes', 'expression']:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        outputs.append(data)
    return outputs

# creates mesh out of vertices and faces
def create_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def load_mesh(data_dir:str, frame_id):
    """
    Loads the meshes from the data_dir

    :param data_dir: The path to the smpl files
    :param frame_ids: The frame id
    :return: Returns a list of the meshes of the frame id
    """
    # loads the smpl model
    body_model = load_model(gender='neutral', model_path='data/smpl_models')

    data = read_smpl(os.path.join(data_dir, str(frame_id).zfill(6) + '.json'))
    # all the meshes in a frame
    frame_meshes = []
    for i in range(len(data)):
        frame = data[i]
        Rh = frame['Rh']
        Th = frame['Th']
        poses = frame['poses']
        shapes = frame['shapes']

        # gets the vertices
        vertices = body_model(poses,
                            shapes,
                            Rh,
                            Th,
                            return_verts=True,
                            return_tensor=False)[0]
        # the mesh
        model = create_mesh(vertices=vertices, faces=body_model.faces)

        frame_meshes.append(model)

    return frame_meshes


def extract_person(trial, anno_frame_ids, point_cloud_dir, labels_dir, max_dist_to_mesh):
    mesh_dir = os.path.join('data', 'smpl_files', trial)

    if not os.path.exists(mesh_dir):
        logger.warn(f"There are no mesh data in {mesh_dir}. Skipping person extraction...")
        return

    for frame_id in tqdm(anno_frame_ids, desc="Extracting people..."):
        pcd_filename = os.path.join(point_cloud_dir, f"{str(frame_id).zfill(4)}_pointcloud.ply")
        label_filename = os.path.join(labels_dir, f"{str(frame_id).zfill(4)}_pointcloud.label")
        point_cloud = o3d.io.read_point_cloud(pcd_filename)

        meshes = load_mesh(mesh_dir, frame_id)

        indices = prune_point_clouds(point_cloud, meshes, max_dist_to_mesh)

        labels.overwrite_labels(label_filename, indices, 3)
