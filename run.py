import __init_paths__
from utils.utils import create_logger
from utils.config import get_cfg_defaults 
from extraction.point_labeler import PointLabeler

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize in 3D')
    parser.add_argument(
        '--cfg', help='configuration file name', type=str, default='./configs/demo.yaml'
    )
    parser.add_argument('-c', '--calibrate', help='Sets the flag to start calibration', action='store_true')
    parser.add_argument('-a', '--annotate', help='Sets the flag to start annotating', action='store_true')
    parser.add_argument('-v', '--visualize', help='Sets the flag to start visualizing', action='store_true')
    parser.add_argument('-m', '--vis_mesh', help='Visualize the point clouds with the mesh', action='store_true')


    args, _ = parser.parse_known_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    return args, cfg


def main():
    logger = create_logger()

    # Define format for logs


    # # Create stdout handler for logging to the console (logs all five levels)

    # # Add both handlers to the logger

    args, labeler_config = parse_args()

    

    pl = PointLabeler(labeler_config)
    if args.calibrate:
        pl.calibrate()
    if args.annotate:
        logger.info("Starting annotating...")
        pl.annotate()
    if args.visualize:
        pl.visualize()
    if args.vis_mesh:
        pl.visualize(vis_meshes=True)

    

if __name__ == "__main__":
    main()