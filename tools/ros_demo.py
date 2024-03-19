import argparse
import glob
from pathlib import Path
import time
import copy
import json
import os

# VISUALIZATION TOOLS
from visual_utils import ros_vis_utils as V

import numpy as np
import torch

# LOCAL IMPORTS
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from datasets.coda import coda_utils

# ROS IMPORTS
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray

from demo import DemoDataset
from queue import Queue

def normalize_color(color):
    normalized_color = [(r / 255, g / 255, b / 255) for r, g, b in color]
    return normalized_color

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml',

                        help='specify the config for demo')
    parser.add_argument('--pc', '--point_cloud_topic', type=str, default='/coda/ouster/points',
                        help='specify the point cloud ros topic name')
    parser.add_argument('--ckpt', type=str, default='../ckpts/coda128_allclass_bestoracle.pth', help='specify the pretrained model')
    parser.add_argument('--ds_rate', type=int, default=5, help='downsample rate for point cloud detections, defaults to every 5 frames')
    parser.add_argument('--viz', type=bool, default=False, help='Publish ROS visualization message')
    parser.add_argument('--debug', type=bool, default=False, help='Display debug information')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

class CodaDetector(Node):
    def __init__(self):
        super().__init__('CODaROSDetector')

        self.args, self.cfg = parse_config()
        self.debug = self.args.debug

        # PC callback control
        self.pc_count = 0
        self.ds_rate = self.args.ds_rate
        self.pc_msg_queue = Queue()

        self.pcdet_logger = common_utils.create_logger()
        self.get_logger().info('-----------------ROS Demo of OpenPCDet-------------------------')

        #1 Fill in dummy dataset to set point features values
        self.dummy_dataset = DemoDataset(
            dataset_cfg=self.cfg.DATA_CONFIG, class_names=self.cfg.CLASS_NAMES, training=False,
            root_path=Path("../data"), ext=".bin", logger=self.pcdet_logger
        )
        self.color_map=normalize_color(coda_utils.BBOX_ID_TO_COLOR)

        #2 Load model
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dummy_dataset)
        self.model.load_params_from_file(filename=self.args.ckpt, logger=self.pcdet_logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        #3 Initialize ROS node/sub/pubs
        self.pc_topic = self.args.pc
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, 1)
        self.bbox_3d_pub = self.create_publisher(MarkerArray, '/coda/bbox_3d', 10)
        self.dets_3d_pub = self.create_publisher(Detection3DArray, '/coda/dets_3d', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/coda/ouster/dt_points', 10)

        #4 Load dummy data to speed up first pass
        dummy_pc = np.random.rand(1000, 3).astype(np.float32)
        dummy_data_dict = V.pcnp_to_datadict(dummy_pc, self.dummy_dataset, frame_id=0)
        pred_dicts, _ = self.model.forward(dummy_data_dict)

        self.get_logger().info("Model initalized...")

    def pc_callback(self, msg):
        if self.debug:
            pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pc_list = list(pc_data)
            pc_np = np.array(pc_list, dtype=np.float32)

            self.get_logger().info("Received point cloud with shape ", pc_np.shape)

        self.pc_count+=1
        if self.pc_count % self.ds_rate == 0:
            if self.pc_msg_queue.qsize() > 1: # Discard non-current frames 
                self.pc_msg_queue.queue.clear()
            self.pc_msg_queue.put(msg)
            self.pc_count = 0

    
        if not self.pc_msg_queue.empty():
            self.pc_msg = self.pc_msg_queue.get()
            
            if self.args.viz:
                V.visualize_3d(self.model, self.dummy_dataset, self.pc_msg, self.bbox_3d_pub, self.color_map, self.pcdet_logger)
            V.publish_3d_dets(self.model, self.dummy_dataset, self.pc_msg, self.dets_3d_pub, self.pcdet_logger)
            # lidar_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)

    coda_detector = CodaDetector()
    rclpy.spin(coda_detector)

    coda_detector.get_logger().info("Demo complete, cleaning up...")
    coda_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()