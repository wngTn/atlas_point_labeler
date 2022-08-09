import os
from extraction.extract_floor import extract_floor
from tqdm import tqdm
import open3d as o3d
import open3d.visualization.gui as gui
import struct
from extraction.extract_person import extract_person
from extraction.extract_wall import extract_wall
import extraction.labels as labelutil
from utils.calibrator import main as cal_main
from glob import glob
import json

import logging
logger = logging.getLogger(__name__)

class PointLabeler:

    def __init__(self, labeler_config):
        
        self.anno_trials = labeler_config.ANNOTATE.TRIALS
        self.anno_frame_ids = list(range(
            labeler_config.ANNOTATE.FRAME_IDS[0],
            labeler_config.ANNOTATE.FRAME_IDS[1],
            labeler_config.ANNOTATE.FRAME_IDS[2]
        ))
        self.max_dist_to_mesh = labeler_config.ANNOTATE.MAX_DIST_TO_MESH

        self.cal_trials = labeler_config.CALIBRATION.TRIALS
        self.x = labeler_config.CALIBRATION._X
        self.y = labeler_config.CALIBRATION._Y
        self.z = labeler_config.CALIBRATION._Z
        self.ground = labeler_config.CALIBRATION.GROUND
        self.voxel_space = labeler_config.CALIBRATION.VOXEL_SPACE
        self.nb_points = labeler_config.CALIBRATION.NB_POINTS
        self.radius = labeler_config.CALIBRATION.RADIUS

        self.vis_trial = labeler_config.VISUALIZE.TRIAL
        self.vis_frame_ids = list(range(
            labeler_config.VISUALIZE.FRAME_IDS[0],
            labeler_config.VISUALIZE.FRAME_IDS[1],
            labeler_config.VISUALIZE.FRAME_IDS[2]
        ))


    
    def visualize(self):
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.show_settings = True

        logger.info(f"Visualizing trial: {self.vis_trial}, with frame ids: {self.vis_frame_ids}")
        data_dir = os.path.join('data', 'label_data', self.vis_trial)

        for frame_id in tqdm(self.vis_frame_ids, desc="Reading the point clouds"):
            point_cloud = o3d.io.read_point_cloud(os.path.join(data_dir, "point_clouds", f"{str(frame_id).zfill(4)}_pointcloud.ply"))
            labels = labelutil.read_labels(os.path.join(data_dir, "labels", f"{str(frame_id).zfill(4)}_pointcloud.label"))

            assert(len(point_cloud.points) == len(labels))

            for i in range(len(labels)):
                if labels[i] == 3:
                    point_cloud.colors[i] = [1, 0, 0]
                elif labels[i] == 2:
                    point_cloud.colors[i] = [0, 1, 0]
                elif labels[i] == 1:
                    point_cloud.colors[i] = [0, 0, 1]

            vis.add_geometry(f'Point cloud of frame {frame_id}', point_cloud)

        
        app.add_window(vis)
        app.run()

    
    def init_output(self):
        """
        Creates the merged point clouds and labels of the calibration trials if necessary
        Sets the point cloud and label directories.
        """
        for trial in self.anno_trials:
            data_dir = os.path.join('data', 'trials', trial)
            if not os.path.exists(data_dir):
                logger.warn(f"There is no data for the trial {trial} in {data_dir}.\nShutting down now...")
                exit(0)

            trial_label_path = os.path.join('data', 'label_data', trial)

            if not os.path.exists(trial_label_path) or not os.path.exists(os.path.join(trial_label_path, 'calibrations.json')):
                logger.warn(f"The calibration file of {trial} are not found in {trial_label_path}. You might\
                    want to calibrate this trial first.")

            if len(glob(trial_label_path + '/**/*')) <= 3 + (len(self.anno_trials)):
                # We have to create the point clouds and labels ourselves
                logger.info(f"There is no data in {trial_label_path}. Writing data now...")
                # Writing all the data ourselves
                point_cloud_output_dir = os.path.join('data', 'label_data', trial, 'point_clouds')
                labels_dir = os.path.join('data', 'label_data', trial, 'labels')

                os.makedirs(point_cloud_output_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                cameras = list(sorted(next(os.walk(data_dir))[1]))
            

                # Create the merged point clouds and labels
                for frame_id in tqdm(self.anno_frame_ids, desc="Writing merged point clouds"):
                    merged_point_cloud = o3d.geometry.PointCloud()
                    file_id = str(frame_id).zfill(4)
                    # get point clouds and origins
                    for cam in cameras:

                        fpath = os.path.join(data_dir, cam, f"{file_id}_pointcloud.ply")
                        if not os.path.exists(fpath):
                            logger.warn("File does not exist: {}".format(fpath))

                        merged_point_cloud += o3d.io.read_point_cloud(fpath)
                    
                    o3d.io.write_point_cloud(pcd_file:=os.path.join(point_cloud_output_dir, f"{file_id}_pointcloud.ply"), merged_point_cloud)

                    # create label file filled with 0s
                    contents = struct.pack('<I', 0) * len(merged_point_cloud.points)

                    with open(label_file:=os.path.join(labels_dir, f"{str(frame_id).zfill(4)}_pointcloud.label"), "bw") as f:
                        f.write(contents)

                    logger.debug(f"Written: {pcd_file} and {label_file}")

    
    def calibrate(self):
        for trial in self.cal_trials:
            first_pcd = create_first_pcd(trial)
            calibration_dict = cal_main(self.x, self.y, self.z, self.ground, first_pcd, self.voxel_space, 
                self.nb_points, self.radius)
            write_configs(trial, calibration_dict)


    def annotate(self):
        self.init_output()

        for trial in self.anno_trials:
            logger.info("Doing annotation for trial: %s" % trial)
            point_cloud_dir = os.path.join('data', 'label_data', trial, 'point_clouds')
            labels_dir = os.path.join('data', 'label_data', trial, 'labels')

            extract_wall(trial, self.anno_frame_ids, point_cloud_dir, labels_dir)
            extract_person(trial, self.anno_frame_ids, point_cloud_dir, labels_dir, self.max_dist_to_mesh)
            extract_floor(trial, self.anno_frame_ids, point_cloud_dir, labels_dir)



def write_configs(trial, calibration_dict):
    os.makedirs(os.path.join('data', 'label_data', trial), exist_ok=True)
    calibration_path = os.path.join('data', 'label_data', trial, 'calibrations.json')
    with open(calibration_path, 'w', encoding='utf-8') as f:
        json.dump(calibration_dict, f, ensure_ascii=False, indent=4)
        logger.info(f"Written: {calibration_path}")



def create_first_pcd(trial):
    trial_path = os.path.join('data', 'trials', trial)

    if not os.path.exists(trial_path):
        logger.warn(f"There is no data for the trial {trial} in {trial_path}.\nShutting down now...")
        exit(0)

    # the ultimate merged pcd of the first point clouds
    merged_pcd = o3d.geometry.PointCloud()

    cameras = next(os.walk(trial_path))[1]

    for camera in cameras:
        cam_path = os.path.join(trial_path, camera)
        path_to_first_pcd = list(sorted(glob(cam_path + '/*.ply')))[0]

        merged_pcd += o3d.io.read_point_cloud(path_to_first_pcd)
        logger.debug(f"Added {path_to_first_pcd} to the merged point cloud")

    return merged_pcd




    
