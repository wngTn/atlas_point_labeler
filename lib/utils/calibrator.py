import open3d as o3d
import numpy as np
import copy
import open3d.visualization.gui as gui
from scipy.spatial.transform import Rotation as rot
from extraction.extract_floor import get_floor

import logging
logger = logging.getLogger(__name__)


init_point_cloud = None # o3d.io.read_point_cloud('point_cloud.ply')
init_p_point_cloud = None
init_point_cloud_wall_mask = o3d.geometry.PointCloud()
point_cloud_floor_mask = o3d.geometry.PointCloud()
mesh_front = o3d.geometry.TriangleMesh()
mesh_rear = o3d.geometry.TriangleMesh()
mesh_left = o3d.geometry.TriangleMesh()
mesh_right = o3d.geometry.TriangleMesh()
rot_matrix = np.eye(4)
min_x = 0
min_y = 0
max_x = 0
max_y = 0
buffer = 0.1
x = 0
y = 0
z = 0
g = 0
rotation_around_z = 0
floor = 0


def update_all(vis):
    global init_point_cloud
    global init_point_cloud_wall_mask
    global mesh_front
    global mesh_rear
    global mesh_right
    global mesh_left
    global point_cloud_floor_mask

    vis.update_geometry(init_point_cloud)
    vis.update_geometry(init_point_cloud_wall_mask)
    vis.update_geometry(mesh_front)
    vis.update_geometry(mesh_rear)
    vis.update_geometry(mesh_right)
    vis.update_geometry(mesh_left)
    vis.update_geometry(point_cloud_floor_mask)


def get_wall_indices(point_cloud, min_x, min_y, max_x, max_y):
    buffer_min = 0.5
    buffer_max = 0.5

    point_cloud_copy = copy.deepcopy(point_cloud)

    only_indices = np.arange(len(point_cloud.points))
    # [ [x, y, z, index], [x, y, z, index], ...]
    #
    #

    max_x_mask = np.logical_and(np.asarray(point_cloud_copy.points)[:, 0] < (max_x + buffer_max), np.asarray(point_cloud_copy.points)[:, 0] > (max_x - buffer_max))

    min_x_mask = np.logical_and(np.asarray(point_cloud_copy.points)[:, 0] < (min_x + buffer_max), np.asarray(point_cloud_copy.points)[:, 0] > (min_x - buffer_min))


    max_y_mask = np.logical_and(np.asarray(point_cloud_copy.points)[:, 1] < (max_y + buffer_max), np.asarray(point_cloud_copy.points)[:, 1] > (max_y - buffer_max))

    min_y_mask = np.logical_and(np.asarray(point_cloud_copy.points)[:, 1] < (min_y + buffer_max), np.asarray(point_cloud_copy.points)[:, 1] > (min_y - buffer_min))


    mask = np.logical_or(np.logical_or(max_x_mask, min_x_mask), np.logical_or(max_y_mask, min_y_mask))

    only_indices = only_indices[mask]

    return only_indices


def get_wall(point_cloud, nb_points, radius):

    cl, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pruned_pcd = point_cloud.select_by_index(ind)

    pcd_points = np.array(pruned_pcd.points)

    # pcd_points = np.array(point_cloud.points)

    min_x = np.min(pcd_points[:, 0])
    min_y = np.min(pcd_points[:, 1])

    max_x = np.max(pcd_points[:, 0])
    max_y = np.max(pcd_points[:, 1])

    return min_x, min_y, max_x, max_y



def get_corner_points(point_cloud):

    pcd_points = np.array(point_cloud.points)

    min_x = pcd_points[np.argmin(pcd_points[:, 0])]
    min_y = pcd_points[np.argmin(pcd_points[:, 1])]

    max_x = pcd_points[np.argmax(pcd_points[:, 0])]
    max_y = pcd_points[np.argmax(pcd_points[:, 1])]

    return min_x, min_y, max_x, max_y


def color_points(_min_x=None, _min_y=None, _max_x=None, _max_y=None):
    # [ [x, y, z, index], [x, y, z, index], ...]
    #
    #
    global init_point_cloud
    global init_point_cloud_wall_mask
    global init_p_point_cloud
    global mesh_front
    global mesh_rear
    global mesh_right
    global mesh_left
    global min_x
    global min_y
    global max_x
    global max_y
    global rot_matrix
    global buffer
    global floor
    global point_cloud_floor_mask

    init_p_point_cloud = init_p_point_cloud.transform(rot_matrix)

    pcd_points = np.array(init_p_point_cloud.points)

    # pcd_points = np.array(point_cloud.points)
    if _min_x == None:
        min_x = np.min(pcd_points[:, 0])
    if _min_y == None:
        min_y = np.min(pcd_points[:, 1])

    if _max_x == None:
        max_x = np.max(pcd_points[:, 0])
    if _max_y == None:
        max_y = np.max(pcd_points[:, 1])

    global x
    global y
    global g
    global z

    triangles = o3d.utility.Vector3iVector(
        np.array([
            [1, 0, 2],
            [2, 3, 1]
        ])
    )

    mesh_front.vertices = o3d.utility.Vector3dVector(
        np.array([
            [-x, max_y, g],
            [-x, max_y, z],
            [x, max_y, g],
            [x, max_y, z]
        ])
    )
    mesh_front.triangles = triangles

    mesh_rear.vertices = o3d.utility.Vector3dVector(
        np.array([
            [-x, min_y, g],
            [-x, min_y, z],
            [x, min_y, g],
            [x, min_y, z]
        ])
    )
    mesh_rear.triangles = triangles

    mesh_left.vertices = o3d.utility.Vector3dVector(
        np.array([
            [min_x, -y, g],
            [min_x, -y, z],
            [min_x, y, g],
            [min_x, y, z]
        ])
    )
    mesh_left.triangles = triangles

    mesh_right.vertices = o3d.utility.Vector3dVector(
        np.array([
            [max_x, -y, g],
            [max_x, -y, z],
            [max_x, y, g],
            [max_x, y, z]
        ])
    )
    mesh_right.triangles = triangles


    only_indices = np.arange(len(init_point_cloud.points))

    init_point_cloud = init_point_cloud.transform(rot_matrix)

    max_x_mask = np.logical_and(np.asarray(init_point_cloud.points)[:, 0] < (max_x + buffer), np.asarray(init_point_cloud.points)[:, 0] > (max_x - buffer))

    min_x_mask = np.logical_and(np.asarray(init_point_cloud.points)[:, 0] < (min_x + buffer), np.asarray(init_point_cloud.points)[:, 0] > (min_x - buffer))


    max_y_mask = np.logical_and(np.asarray(init_point_cloud.points)[:, 1] < (max_y + buffer), np.asarray(init_point_cloud.points)[:, 1] > (max_y - buffer))

    min_y_mask = np.logical_and(np.asarray(init_point_cloud.points)[:, 1] < (min_y + buffer), np.asarray(init_point_cloud.points)[:, 1] > (min_y - buffer))

    mask = np.logical_or(np.logical_or(max_x_mask, min_x_mask), np.logical_or(max_y_mask, min_y_mask))

    only_indices = only_indices[mask]

    temp_mask = init_point_cloud.select_by_index(only_indices)

    init_point_cloud_wall_mask.points = temp_mask.points

    init_point_cloud_wall_mask.paint_uniform_color([0.8, 0, 0])


    point_cloud_floor_mask.points = init_point_cloud.select_by_index(
        np.intersect1d(np.where(np.asarray(init_point_cloud.points)[:, 2] <= floor + 0.06)[0], np.where(np.asarray(init_point_cloud.points)[:, 2] >= floor - 0.06)[0], assume_unique=True)
    ).points

    point_cloud_floor_mask.paint_uniform_color([0, 0.7, 0])



def key_callback_floor_up(vis):
    global floor
    global rot_matrix
    rot_matrix = np.eye(4)

    floor += 0.01
    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    vis.update_geometry(point_cloud_floor_mask)
    #print(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    update_all(vis)
    vis.poll_events()
    vis.update_renderer()
    logger.info(f"Increased floor, floor now at {floor}")


def key_callback_floor_down(vis):
    global floor
    global rot_matrix
    rot_matrix = np.eye(4)

    floor -= 0.01
    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    vis.update_geometry(point_cloud_floor_mask)
    #print(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    update_all(vis)
    vis.poll_events()
    vis.update_renderer()
    logger.info(f"Decreased floor, floor now at {floor}")


def key_callback_up(vis):
    global buffer
    global rot_matrix

    rot_matrix = np.eye(4)

    buffer += 0.02
    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #print(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Increased Buffer, buffer now at {buffer}")


def key_callback_down(vis):
    global buffer
    global rot_matrix

    rot_matrix = np.eye(4)

    buffer -= 0.02
    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #print(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Decreased Buffer, buffer now at {buffer}")


def key_callback_G(vis):
    global rot_matrix
    global min_x

    rot_matrix = np.eye(4)
    min_x += 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #logger.debug(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved left wall medial, now at x={min_x}")


def key_callback_F(vis):
    global rot_matrix
    global min_x

    rot_matrix = np.eye(4)
    min_x -= 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #logger.debug(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved left wall lateral, now at x={min_x}")


def key_callback_T(vis):
    global rot_matrix
    global max_x
    rot_matrix = np.eye(4)

    max_x += 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #logger.debug(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved right wall lateral, now at x={max_x}")



def key_callback_R(vis):
    global rot_matrix
    global max_x
    rot_matrix = np.eye(4)

    max_x -= 0.01
    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)

    update_all(vis)
    #logger.debug(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved right wall medial, now at x={max_x}")


def key_callback_D(vis):
    global rot_matrix
    global min_y
    rot_matrix = np.eye(4)

    min_y -= 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #logger.debug(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved rear wall distal, now at y={min_y}")


def key_callback_E(vis):
    global rot_matrix
    global min_y
    rot_matrix = np.eye(4)

    min_y += 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    #logger.debug(f"First point of wall_mask: {np.asarray(init_point_cloud_wall_mask.points)[:5]}")
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved rear wall proximal, now at y={min_y}")


def key_callback_W(vis):
    global rot_matrix
    global max_y
    rot_matrix = np.eye(4)

    max_y += 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved frontal wall distal, now at y={max_y}")

def key_callback_S(vis):
    global rot_matrix
    global max_y
    rot_matrix = np.eye(4)

    max_y -= 0.01

    color_points(_min_x=1, _min_y=1, _max_x=1, _max_y=1)
    update_all(vis)
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Moved frontal wall proximal, now at y={max_y}")


def key_callback_left(vis):

    global init_point_cloud
    global rot_matrix
    global init_point_cloud_wall_mask
    global rotation_around_z

    r = rot.from_euler('z', -1, degrees=True)
    rotation_around_z -= 1



    r_left = np.eye(4)
    r_left[:3, :3] = r.as_matrix()

    rot_matrix = r_left
    # init_point_cloud.transform(r_left)

    color_points()

    update_all(vis)
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Rotated around z left by one degree. Rotation is now {rotation_around_z}")

def key_callback_exit(vis):
    logger.info("You just closed the calibrator. Nothing happens")
    exit(0)


def key_callback_right(vis):

    global init_point_cloud
    global rot_matrix
    global init_point_cloud_wall_mask
    global rotation_around_z

    r = rot.from_euler('z', 1, degrees=True)
    rotation_around_z += 1


    r_left = np.eye(4)
    r_left[:3, :3] = r.as_matrix()

    rot_matrix = r_left

    color_points()

    update_all(vis)
    vis.poll_events()
    vis.update_renderer()
    logger.debug(f"Rotated around z right. Rotation is now {rotation_around_z}")


def key_callback_enter(vis):
    vis.destroy_window()



def main(_x, _y, _z, ground, init_pcd, voxel_space, nb_points, radius):
    """
    _summary_

    :param short_side: Approximation of the short side in meters
    :param long_side: Approximation of the long side in meters
    :param ground: Approximation of the height in meters
    """

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    global x
    global y
    global g
    global z
    x = _x / 2
    y = _y / 2

    g = ground # approximation of the ground
    z = _z + g

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-x, -y, g), max_bound=(x, y, z))
    # sets the box to purple
    bbox.color = [0.627, 0.125, 0.941]

    # assigns the initial point cloud
    global init_point_cloud
    init_point_cloud = init_pcd

    logger.info("Approximating the floor...")
    global floor
    floor = get_floor(init_pcd)
    logger.info(f"Floor approximated at {floor}")

    # removes outlier of the initial point cloud
    logger.info("Removing outliers from the point cloud...")
    voxel_down_init = init_pcd.voxel_down_sample(voxel_space)
    _, ind = voxel_down_init.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pruned_pcd = voxel_down_init.select_by_index(ind)
    logger.info("Removed outliers from the point cloud!")

    # assigns the initial pruned point cloud
    global init_p_point_cloud
    init_p_point_cloud = pruned_pcd

    color_points()

    vis.add_geometry(init_point_cloud)
    vis.add_geometry(init_point_cloud_wall_mask)
    vis.add_geometry(point_cloud_floor_mask)
    vis.add_geometry(mesh_front)
    vis.add_geometry(mesh_rear)
    vis.add_geometry(mesh_left)
    vis.add_geometry(mesh_right)
    vis.add_geometry(bbox)

    # left key, rotate left
    vis.register_key_callback(263, key_callback_left)
    # right key, rotate right
    vis.register_key_callback(262, key_callback_right)
    # up, increase buffer by 0.02
    vis.register_key_callback(265, key_callback_up)
    # down, decrease buffer by 0.025
    vis.register_key_callback(264, key_callback_down)

    # W, front wall goes distal
    vis.register_key_callback(87, key_callback_W)

    # S, front wall goes proximal
    vis.register_key_callback(83, key_callback_S)

    # E, rear wall goes proximal
    vis.register_key_callback(69, key_callback_E)
    # D, rear wall goes distal
    vis.register_key_callback(68, key_callback_D)

    # R, right wall goes medial
    vis.register_key_callback(82, key_callback_R)
    # T, right wall goes lateral
    vis.register_key_callback(84, key_callback_T)

    # F, left wall goes medial
    vis.register_key_callback(70, key_callback_F)
    # G, left wall goes lateral
    vis.register_key_callback(71, key_callback_G)

    # Comma, floor will go down
    vis.register_key_callback(44, key_callback_floor_down)

    # Period, floor will go up
    vis.register_key_callback(46, key_callback_floor_up)

    vis.register_key_callback(257, key_callback_enter)


    vis.register_key_callback(66, key_callback_exit)


    vis.get_render_option().mesh_show_back_face = True


    vis.run()

    logger.debug(f"\n\tRotation is: {rotation_around_z}\n\
        min_x at: {min_x}\n\
        min_y at: {min_y}\n\
        max_x at: {max_x}\n\
        max_y at: {max_y}\n\
        buffer at: {buffer}\n\
        floor at: {floor}")

    calibration_dict = {
        "rot_around_z" : rotation_around_z,
        "min_x" : min_x,
        "min_y" : min_y,
        "max_x" : max_x,
        "max_y" : max_y,
        "buffer" : buffer,
        "floor" : floor
    }

    return calibration_dict