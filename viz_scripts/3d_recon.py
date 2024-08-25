import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import copy

# Copyright (c) OpenMMLab. All rights reserved.
import logging
import math
import mimetypes
import time
from argparse import ArgumentParser
from typing import List

import cv2
import json_tricks as json
import matplotlib.pyplot as plt
import mediapipe as mp
import mmcv
import mmengine
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample, merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from mmpose.visualization import Pose3dLocalVisualizer

from ultralytics import YOLO  # 将YOLOv8导入到该py文件中
from ycy_utils.common_utils import seed_everything
from ycy_utils.recon_helpers import setup_camera
from ycy_utils.slam_external import build_rotation
from ycy_utils.slam_helpers import get_depth_and_silhouette

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from rtmpose3d import *  # noqa: F401, F403

MIN_DEPTH = 300  # 20mm
MAX_DEPTH = 6000  # 10000mm
FORCE_LOOP = True
NUM_STEP = 2.5

alpha = 0.6

n_l = 2
n_c = 2 * n_l + 1
num = n_c**3
fps = 1

MAX_ANGLE = 10
NUM_ANGLE = 101
MAX_TRANSLATE = 0.2
MAX_DIST = 0.1

pose_keypoints_id = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

# angle_list = [angle for angle in range(-MAX_ANGLE ,MAX_ANGLE + 1 , 1)]+[angle for angle in range(MAX_ANGLE ,-MAX_ANGLE - 1, -1)]
angle_list = np.append(
    np.linspace(-MAX_ANGLE / 2, MAX_ANGLE, num=NUM_ANGLE),
    np.linspace(MAX_ANGLE, -MAX_ANGLE / 2, num=NUM_ANGLE),
)
translate_list = np.append(
    np.linspace(-MAX_TRANSLATE, MAX_TRANSLATE, num=NUM_ANGLE),
    np.linspace(MAX_TRANSLATE, -MAX_TRANSLATE, num=NUM_ANGLE),
)
dist_list = np.append(
    np.linspace(-MAX_DIST, 0, num=NUM_ANGLE),
    np.linspace(0, -MAX_DIST, num=NUM_ANGLE),
)


# skeleton = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
skeleton = [
    (0, 1),
    (0, 2),
    (0, 5),
    (0, 6),
    (1, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (5, 11),
    (6, 8),
    (6, 12),
    (7, 9),
    (8, 10),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

# pose_link = np.array([[0, 1], [1, 7], [7, 6], [6, 0], [1, 3], [3, 5], [0, 2], [2, 4], [6, 8], [8, 10], [7, 9], [9, 11]])

# use_lines = []
#
# for i in range(num):
#     use_lines.extend([[i + 0 * num, i + 1 * num], [i + 1 * num, i + 7 * num], [i + 7 * num, i + 6 * num],
#                       [i + 6 * num, i + 0 * num], [i + 1 * num, i + 3 * num], [i + 3 * num, i + 5 * num],
#                       [i + 0 * num, i + 2 * num], [i + 2 * num, i + 4 * num], [i + 6 * num, i + 8 * num],
#                       [i + 8 * num, i + 10 * num],[i + 7 * num, i + 9 * num], [i + 9 * num, i + 11 * num]])
#
# pose_link = np.array(use_lines)

# init_depth = 5 * torch.ones((12,), device="cuda:0", dtype=torch.float32)
# init_points = 100 *torch.ones((12, 2), device="cuda:0", dtype=torch.float32)

# init_points = [[205.,  20.],
#         [ 85.,  24.],
#         [240., 105.],
#         [ 57., 109.],
#         [266., 180.],
#         [110.,  90.],
#         [210., 190.],
#         [142., 204.],
#         [233., 307.],
#         [156., 327.],
#         [255., 391.],
#         [165., 421.]]
init_points = np.ones((12 * num, 2)).tolist()

# init_depth = [1.6240, 1.6190, 1.0450, 1.0440, 1.0530, 0.9610, 1.0700, 1.1000, 1.8490,
#         1.8900, 1.4620, 1.2900]
init_depth = np.ones(12 * num).tolist()

k_c2w = np.array(
    [
        [0.98713, -0.06007, -0.14818, 0.1619],
        [0.06602, 0.99718, 0.03559, -0.13114],
        [0.14562, -0.04492, 0.98832, 0.49675],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

i_c2w = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class PersonGeometry:
    def __init__(self, show_points, key=False):
        neck_point = (show_points[5] + show_points[6]) / 2.0
        crotch_point = (show_points[11] + show_points[12]) / 2.0
        body_point = (neck_point + crotch_point) / 2.0
        croch2neck = neck_point - crotch_point
        body_height = np.linalg.norm(croch2neck)
        croch2neck_norm = croch2neck / body_height
        # body_radius = (
        #     max(
        #         np.linalg.norm(show_points[5] - show_points[6]),
        #         np.linalg.norm(show_points[11] - show_points[12]),
        #     )
        #     / 2.0
        # )
        body_radius = 0.1
        head_radius = body_radius / 2.0
        sphere_radius = head_radius / 2.0
        head_point = neck_point + head_radius * croch2neck_norm

        if key:
            body_color = [0.9, 0.9, 0.1]
        else:
            body_color = [0.1, 0.9, 0.9]

        # self.head = o3d.geometry.TriangleMesh.create_sphere(radius=head_radius)
        # self.head.compute_vertex_normals()
        # self.head.paint_uniform_color([0.9, 0.1, 0.1])
        # translation_matrix[:3, 3] = head_point
        # self.head.transform(translation_matrix)
        self.head = self.create_sphere(head_point, head_radius, [0.9, 0.1, 0.1])
        self.l_shoulder = self.create_sphere(show_points[5], sphere_radius)
        self.r_shoulder = self.create_sphere(show_points[6], sphere_radius)
        self.l_elbow = self.create_sphere(show_points[7], sphere_radius)
        self.r_elbow = self.create_sphere(show_points[8], sphere_radius)
        self.l_wrist = self.create_sphere(show_points[9], sphere_radius)
        self.r_wrist = self.create_sphere(show_points[10], sphere_radius)
        self.l_hip = self.create_sphere(show_points[11], sphere_radius)
        self.r_hip = self.create_sphere(show_points[12], sphere_radius)
        self.l_knee = self.create_sphere(show_points[13], sphere_radius)
        self.r_knee = self.create_sphere(show_points[14], sphere_radius)
        self.l_ankle = self.create_sphere(show_points[15], sphere_radius)
        self.r_ankle = self.create_sphere(show_points[16], sphere_radius)

        # self.body = o3d.geometry.TriangleMesh.create_cylinder(
        #     radius=body_radius, height=body_height
        # )
        # self.body.compute_vertex_normals()
        # self.body.paint_uniform_color([0.1, 0.1, 0.9])
        # n = np.cross(
        #     init_norm, croch2neck_norm
        # )  # 计算旋转轴 n，这里我们使用叉积找到垂直于 init 和 c2n 的向量
        # n_norm = n / np.linalg.norm(n)  # 归一化旋转轴
        # theta = np.arccos(np.dot(init_norm, croch2neck_norm))  # 计算旋转角度
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        #     n_norm, theta
        # )
        # translation_matrix[:3, 3] = body_point
        # transformation_matrix = np.dot(
        #     translation_matrix, rotation_matrix
        # )  # 组合旋转和平移矩阵
        # self.body.transform(transformation_matrix)  # 应用变换矩阵

        self.body = self.create_cylinder(
            neck_point, crotch_point, body_radius, color=body_color
        )
        self.l_hindarm = self.create_cylinder(
            show_points[5], show_points[7], sphere_radius
        )
        self.r_hindarm = self.create_cylinder(
            show_points[6], show_points[8], sphere_radius
        )
        self.l_forearm = self.create_cylinder(
            show_points[7], show_points[9], sphere_radius
        )
        self.r_forearm = self.create_cylinder(
            show_points[8], show_points[10], sphere_radius
        )
        self.l_thigh = self.create_cylinder(
            show_points[11], show_points[13], sphere_radius
        )
        self.r_thigh = self.create_cylinder(
            show_points[12], show_points[14], sphere_radius
        )
        self.l_calf = self.create_cylinder(
            show_points[13], show_points[15], sphere_radius
        )
        self.r_calf = self.create_cylinder(
            show_points[14], show_points[16], sphere_radius
        )

    def create_sphere(self, center, radius, color=[0.1, 0.9, 0.1]):
        translation_matrix = np.eye(4)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius,resolution=100)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        translation_matrix[:3, 3] = center
        sphere.transform(translation_matrix)
        return sphere

    def create_cylinder(self, end, start, radius, color=[0.1, 0.1, 0.9]):
        translation_matrix = np.eye(4)
        init_norm = np.array([0, 0, 1])  # 定义初始与 Z 轴平行的向量

        center = (start + end) / 2.0
        target = end - start

        target_norm = target / np.linalg.norm(target)

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=np.linalg.norm(target)+0.01, resolution=100
        )
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)
        n = np.cross(
            init_norm, target_norm
        )  # 计算旋转轴 n，这里我们使用叉积找到垂直于 init 和 c2n 的向量
        n_norm = n / np.linalg.norm(n)  # 归一化旋转轴
        n_norm = n_norm.reshape(3, 1)
        theta = np.arccos(np.dot(init_norm, target_norm))  # 计算旋转角度
        # print("n_norm", n_norm.dtype)
        # print("theta", theta)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            n_norm * theta
        )
        translation_matrix[:3, :3] = rotation_matrix
        translation_matrix[:3, 3] = center
        # transformation_matrix = np.dot(
        #     translation_matrix, rotation_matrix)  # 组合旋转和平移矩阵
        cylinder.transform(translation_matrix)  # 应用变换矩阵
        return cylinder

    def add_geometry(self, visualization):
        visualization.add_geometry(self.head)
        visualization.add_geometry(self.body)
        visualization.add_geometry(self.l_ankle)
        visualization.add_geometry(self.l_calf)
        visualization.add_geometry(self.l_elbow)
        visualization.add_geometry(self.l_forearm)
        visualization.add_geometry(self.l_hindarm)
        visualization.add_geometry(self.l_hip)
        visualization.add_geometry(self.l_knee)
        visualization.add_geometry(self.l_shoulder)
        visualization.add_geometry(self.l_thigh)
        visualization.add_geometry(self.l_wrist)
        visualization.add_geometry(self.r_ankle)
        visualization.add_geometry(self.r_calf)
        visualization.add_geometry(self.r_elbow)
        visualization.add_geometry(self.r_forearm)
        visualization.add_geometry(self.r_hindarm)
        visualization.add_geometry(self.r_hip)
        visualization.add_geometry(self.r_knee)
        visualization.add_geometry(self.r_shoulder)
        visualization.add_geometry(self.r_thigh)
        visualization.add_geometry(self.r_wrist)
        # return visualization

    def remove_geometry(self, visualization):
        visualization.remove_geometry(self.head)
        visualization.remove_geometry(self.body)
        visualization.remove_geometry(self.l_ankle)
        visualization.remove_geometry(self.l_calf)
        visualization.remove_geometry(self.l_elbow)
        visualization.remove_geometry(self.l_forearm)
        visualization.remove_geometry(self.l_hindarm)
        visualization.remove_geometry(self.l_hip)
        visualization.remove_geometry(self.l_knee)
        visualization.remove_geometry(self.l_shoulder)
        visualization.remove_geometry(self.l_thigh)
        visualization.remove_geometry(self.l_wrist)
        visualization.remove_geometry(self.r_ankle)
        visualization.remove_geometry(self.r_calf)
        visualization.remove_geometry(self.r_elbow)
        visualization.remove_geometry(self.r_forearm)
        visualization.remove_geometry(self.r_hindarm)
        visualization.remove_geometry(self.r_hip)
        visualization.remove_geometry(self.r_knee)
        visualization.remove_geometry(self.r_shoulder)
        visualization.remove_geometry(self.r_thigh)
        visualization.remove_geometry(self.r_wrist)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    # parser.add_argument("det_config", help="Config file for detection")
    # parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument(
        "pose3d_estimator_config",
        type=str,
        default=None,
        help="Config file for the 3D pose estimator",
    )
    parser.add_argument(
        "pose3d_estimator_checkpoint",
        type=str,
        default=None,
        help="Checkpoint file for the 3D pose estimator",
    )
    parser.add_argument("--input", type=str, default="", help="Video path")
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Whether to show visualizations",
    )
    parser.add_argument(
        "--disable-rebase-keypoint",
        action="store_true",
        default=False,
        help="Whether to disable rebasing the predicted 3D pose so its "
        "lowest keypoint has a height of 0 (landing on the ground). Rebase "
        "is useful for visualization when the model do not predict the "
        "global position of the 3D pose.",
    )
    parser.add_argument(
        "--disable-norm-pose-2d",
        action="store_true",
        default=False,
        help="Whether to scale the bbox (along with the 2D pose) to the "
        "average bbox scale of the dataset, and move the bbox (along with the "
        "2D pose) to the average bbox center of the dataset. This is useful "
        "when bbox is small, especially in multi-person scenarios.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="The number of 3D poses to be visualized in every frame. If "
        "less than 0, it will be set to the number of pose results in the "
        "first frame.",
    )
    # parser.add_argument(
    #     "--output-root",
    #     type=str,
    #     default="",
    #     help="Root of the output video file. "
    #     "Default not saving the visualization video.",
    # )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="Whether to save predicted results",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=0,
        help="Category id for bounding box detection model",
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.6, help="Bounding box score threshold"
    )
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument(
        "--use-oks-tracking", action="store_true", help="Using OKS tracking"
    )
    parser.add_argument(
        "--tracking-thr", type=float, default=0.3, help="Tracking threshold"
    )
    parser.add_argument(
        "--show-interval", type=float, default=0.001, help="Sleep seconds per frame"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    # parser.add_argument(
    #     "--online",
    #     action="store_true",
    #     default=False,
    #     help="Inference mode. If set to True, can not use future frame"
    #     "information when using multi frames for inference in the 2D pose"
    #     "detection stage. Default: False.",
    # )
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="confidence threshold"
    )
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detection")
    parser.add_argument(
        "--tracker",
        type=str,
        default="my.yaml",
        help="yaml file for the 3D pose estimator",
    )

    args = parser.parse_args()
    return args


def process_one_image(
    args,
    detector,
    frame: np.ndarray,
    frame_idx: int,
    pose_estimator,
    pose_est_results_last: List[PoseDataSample],
    pose_est_results_list: List[List[PoseDataSample]],
    next_id: int,
    visualize_frame: np.ndarray,
    visualizer: Pose3dLocalVisualizer,
):
    """Visualize detected and predicted keypoints of one image.

    Pipeline of this function:

                              frame
                                |
                                V
                        +-----------------+
                        |     detector    |
                        +-----------------+
                                |  det_result
                                V
                        +-----------------+
                        |  pose_estimator |
                        +-----------------+
                                |  pose_est_results
                                V
                       +-----------------+
                       | post-processing |
                       +-----------------+
                                |  pred_3d_data_samples
                                V
                         +------------+
                         | visualizer |
                         +------------+

    Args:
        args (Argument): Custom command-line arguments.
        detector (mmdet.BaseDetector): The mmdet detector.
        frame (np.ndarray): The image frame read from input image or video.
        frame_idx (int): The index of current frame.
        pose_estimator (TopdownPoseEstimator): The pose estimator for 2d pose.
        pose_est_results_last (list(PoseDataSample)): The results of pose
            estimation from the last frame for tracking instances.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            pose estimation results converted by
            ``convert_keypoint_definition`` from previous frames. In
            pose-lifting stage it is used to obtain the 2d estimation sequence.
        next_id (int): The next track id to be used.
        pose_lifter (PoseLifter): The pose-lifter for estimating 3d pose.
        visualize_frame (np.ndarray): The image for drawing the results on.
        visualizer (Visualizer): The visualizer for visualizing the 2d and 3d
            pose estimation results.

    Returns:
        pose_est_results (list(PoseDataSample)): The pose estimation result of
            the current frame.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            converted pose estimation results until the current frame.
        pred_3d_instances (InstanceData): The result of pose-lifting.
            Specifically, the predicted keypoints and scores are saved at
            ``pred_3d_instances.keypoints`` and
            ``pred_3d_instances.keypoint_scores``.
        next_id (int): The next track id to be used.
    """
    # pose_dataset = pose_estimator.cfg.test_dataloader.dataset
    pose_det_dataset_name = pose_estimator.dataset_meta["dataset_name"]

    # # First stage: conduct 2D pose detection in a Topdown manner
    # # use detector to obtain person bounding boxes
    # det_result = inference_detector(detector, frame)
    # pred_instance = det_result.pred_instances.cpu().numpy()

    # # filter out the person instances with category and bbox threshold
    # # e.g. 0 for person in COCO
    # bboxes = pred_instance.bboxes
    # bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
    #                                pred_instance.scores > args.bbox_thr)]

    yolo_results = detector.track(
        source=frame,
        conf=args.conf_thres,
        iou=args.iou_thres,
        max_det=args.max_det,
        classes=args.det_cat_id,
        tracker=args.tracker,
        device=args.device,
        verbose=False,
        retina_masks=True,
        persist=True,
    )
    if yolo_results[0].boxes.xyxy is None or yolo_results[0].boxes.id is None:
        bboxes = []
        bboxes_id = []
        bboxes_xywh = []
    else:
        bboxes = yolo_results[0].boxes.xyxy.tolist()
        bboxes_id = yolo_results[0].boxes.id.tolist()
        bboxes_xywh = yolo_results[0].boxes.xywh.tolist()

    plotted_image = yolo_results[0].plot(
        conf=True,
        line_width=1,
        font_size=1,
        # font=Arial.ttf,
        labels=True,
        boxes=True,
    )

    # estimate pose results for current image
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)
    data_samples = merge_data_samples(pose_est_results)

    # post-processing
    for idx, pose_est_result in enumerate(pose_est_results):
        # pose_est_result.track_id = pose_est_results[idx].get('track_id', 1e4)
        if idx < len(bboxes_id):
            pose_est_result.track_id = bboxes_id[idx]
        else:
            pose_est_result.track_id = pose_est_results[idx].get("track_id", 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_results[idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = -keypoints[..., [0, 2, 1]]

        # rebase height (z-axis)
        if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)

        pose_est_results[idx].pred_instances.keypoints = keypoints

    pose_est_results = sorted(pose_est_results, key=lambda x: x.get("track_id", 1e4))

    pred_3d_data_samples = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get("pred_instances", None)

    if args.num_instances < 0:
        args.num_instances = len(pose_est_results)

    # # Visualization
    # if visualizer is not None:
    #     visualizer.add_datasample(
    #         'result',
    #         visualize_frame,
    #         data_sample=pred_3d_data_samples,
    #         det_data_sample=pred_3d_data_samples,
    #         draw_gt=False,
    #         draw_2d=True,
    #         dataset_2d=pose_det_dataset_name,
    #         dataset_3d=pose_det_dataset_name,
    #         show=args.show,
    #         draw_bbox=True,
    #         kpt_thr=args.kpt_thr,
    #         convert_keypoint=False,
    #         axis_limit=400,
    #         axis_azimuth=70,
    #         axis_elev=15,
    #         num_instances=args.num_instances,
    #         wait_time=args.show_interval)

    return pose_est_results, pose_est_results_list, pred_3d_instances, next_id, bboxes_xywh, plotted_image


def load_camera(cfg, scene_path):
    """_summary_

    Args:
        cfg (_type_): _description_
        scene_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params["org_width"]
    org_height = params["org_height"]
    w2c = params["w2c"]
    intrinsics = params["intrinsics"]
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg["viz_w"] / org_width
    k[1, :] *= cfg["viz_h"] / org_height
    return w2c, k


def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {
        k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()
    }
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [
        k
        for k in all_params.keys()
        if k
        not in [
            "org_width",
            "org_height",
            "w2c",
            "intrinsics",
            "gt_w2c_all_frames",
            "cam_unnorm_rots",
            "cam_trans",
            "keyframe_time_indices",
        ]
    ]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params["cam_unnorm_rots"].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., t_i])
        cam_tran = params["cam_trans"][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    # Check if Gaussians are Isotropic or Anisotropic
    if params["log_scales"].shape[-1] == 1:
        log_scales = torch.tile(params["log_scales"], (1, 3))
    else:
        log_scales = params["log_scales"]

    rendervar = {
        "means3D": params["means3D"],
        "colors_precomp": params["rgb_colors"],
        "rotations": torch.nn.functional.normalize(params["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(params["means3D"], device="cuda"),
    }
    depth_rendervar = {
        "means3D": params["means3D"],
        "colors_precomp": get_depth_and_silhouette(params["means3D"], first_frame_w2c),
        "rotations": torch.nn.functional.normalize(params["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(params["means3D"], device="cuda"),
    }
    return rendervar, depth_rendervar, all_w2cs


def pose_link(num_people):
    pose_links = []
    for j in range(num_people):
        n_p = 12 * num * j
        for i in range(num):
            pose_links.extend(
                [
                    [i + 0 * num + n_p, i + 1 * num + n_p],
                    [i + 1 * num + n_p, i + 7 * num + n_p],
                    [i + 7 * num + n_p, i + 6 * num + n_p],
                    [i + 6 * num + n_p, i + 0 * num + n_p],
                    [i + 1 * num + n_p, i + 3 * num + n_p],
                    [i + 3 * num + n_p, i + 5 * num + n_p],
                    [i + 0 * num + n_p, i + 2 * num + n_p],
                    [i + 2 * num + n_p, i + 4 * num + n_p],
                    [i + 6 * num + n_p, i + 8 * num + n_p],
                    [i + 8 * num + n_p, i + 10 * num + n_p],
                    [i + 7 * num + n_p, i + 9 * num + n_p],
                    [i + 9 * num + n_p, i + 11 * num + n_p],
                ]
            )
    # pose_links = np.array(use_lines)
    return pose_links


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pts, np.float64)
        )
        lineset.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(cols, np.float64)
        )
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(
            np.ascontiguousarray(line_indices, np.int32)
        )
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        cam = setup_camera(
            cfg["viz_w"], cfg["viz_h"], k, w2c, cfg["viz_near"], cfg["viz_far"]
        )
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered,
        )
        (
            im,
            _,
            depth,
            _,
            _,
        ) = Renderer(
            raster_settings=white_bg_cam
        )(**timestep_data)
        (
            depth_sil,
            _,
            _,
            _,
            _,
        ) = Renderer(
            raster_settings=cam
        )(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    # Colorize point cloud
    if cfg["render_mode"] == "depth":
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap("jet")
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array(
            [1.0, 1.0, 1.0]
        )
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def ycy_rgbd2pcd(
    color, depth, w2c, intrinsics, cfg, s_points, s_depth, k_c2w, use_lines
):  # lines
    width, height = color.shape[2], color.shape[1]
    lines = np.array(use_lines)

    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    s_xx = s_points[:, 0]
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    s_yy = s_points[:, 1]
    xx = (xx - CX) / FX
    s_xx = (s_xx - CX) / FX

    yy = (yy - CY) / FY
    s_yy = (s_yy - CY) / FY

    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # print((k_xx * k_depth).shape)

    # print(k_yy.shape)

    s_pts_cam = torch.stack((s_xx * s_depth, s_yy * s_depth, s_depth), dim=-1)
    # print("k_pts_cam", k_pts_cam.dtype)
    s_pix_ones = torch.ones(s_points.shape[0], 1).cuda().float()
    s_pts4 = torch.cat((s_pts_cam, s_pix_ones), dim=1)
    # print("k_pts4", k_pts4.dtype)
    s_pts = (k_c2w @ s_pts4.T).T[:, :3]

    # print("s_pts", s_pts)

    # 绘制颜色
    green_color = torch.tensor([0.0, 1.0, 0.0], device="cuda:0")
    spoints_cols = green_color.repeat(s_points.shape[0], 1)
    red_color = torch.tensor([1.0, 0.0, 0.0], device="cuda:0")
    klines_cols = red_color.repeat(lines.shape[0], 1)

    # 绘制线条
    # lines_pcd = o3d.geometry.LineSet()
    # print(lines.shape)
    pose_lineset = o3d.utility.Vector2iVector(np.ascontiguousarray(lines, np.int32))
    pose_colors = o3d.utility.Vector3dVector(
        klines_cols.contiguous().double().cpu().numpy()
    )  # 线条颜色
    pose_points = o3d.utility.Vector3dVector(s_pts.contiguous().double().cpu().numpy())

    pts = torch.cat((pts, s_pts), 0)

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    # Colorize point cloud
    if cfg["render_mode"] == "depth":
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap("jet")
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array(
            [1.0, 1.0, 1.0]
        )
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = torch.cat((cols, spoints_cols), 0)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols, pose_lineset, pose_colors, pose_points


def get_yolov5(dir, model):
    # Model
    # 加载模型文件
    # yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # 加载轻量级的模型文件
    yolo_model = torch.hub.load(dir, model, pretrained=True)

    # 检查是否有可用的GPU设备
    if torch.cuda.is_available():
        # 将yolo_model加载到GPU设备上
        yolo_model = yolo_model.to("cuda")
    else:
        print("GPU device not found. Using CPU instead.")

    # since we are only interested in detecting a person
    yolo_model.classes = [0]

    return yolo_model


def get_pixel_depth(depth: np.ndarray, x, y, windowsize: int = 1):
    total_depth = 0
    ix = max(min(x, depth.shape[1] - 1 - windowsize), windowsize)
    iy = max(min(y, depth.shape[0] - 1 - windowsize), windowsize)
    total_depth = []
    for i in range(-windowsize, windowsize + 1):
        for j in range(-windowsize, windowsize + 1):
            if depth[iy + i, ix + j] > MIN_DEPTH and depth[iy + i, ix + j] < MAX_DEPTH:
                total_depth.append(depth[iy + i, ix + j])
    if len(total_depth):
        np_depth = np.array(total_depth)
        pixel_depth = np.median(np_depth)
    else:
        pixel_depth = depth[y, x]
    return pixel_depth


def get_show(
    img: np.ndarray,
    depth: np.ndarray,
    pred_3d_instances: InstanceData,
    intrinsics: np.ndarray,
    track_id,
    bboxes_xywh,
    kpt_thr: float = 0.3,
):
    show_points = []
    pose_links = []
    point_num = 0
    key_errors = []
    h, w, c = img.shape  # 返回图片的(高,宽,位深)
    t_points = pred_3d_instances.get(
        "transformed_keypoints", pred_3d_instances.keypoints
    )
    person = t_points.shape[0]
    depth_mean = []
    depth_id = []

    # print("person", person)
    scores = pred_3d_instances.get(
        "keypoints_scores", np.ones(pred_3d_instances.keypoints.shape[:-1])
    )
    visibles = pred_3d_instances.get(
        "keypoints_visible", np.ones(pred_3d_instances.keypoints.shape[:-1])
    )

    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    for i in range(person):
        temp_points = []
        valid = np.zeros((17,), dtype=int)
        print("xywh_shape", bboxes_xywh)
        l_hip_x = int(max(min(t_points[i, 11, 0], w - 1), 0))
        l_hip_y = int(max(min(t_points[i, 11, 1], h - 1), 0))
        r_hip_x = int(max(min(t_points[i, 12, 0], w - 1), 0))
        r_hip_y = int(max(min(t_points[i, 12, 1], h - 1), 0))
        center_depth = (
            get_pixel_depth(depth, l_hip_x, l_hip_y)
            + get_pixel_depth(depth, r_hip_x, r_hip_y)
        ) / 2000.0
        # center_depth = (depth[int(l_hip_y), int(l_hip_x)] + depth[int(r_hip_y), int(r_hip_x)]) / 2000.0
        # print("center-depth", center_depth)
        if center_depth > 1e-2:
            cd = 2 * center_depth / (intrinsics[0, 0] + intrinsics[1, 1])
            for j in range(17):  # coco17
                score = scores[i, j]
                visible = visibles[i, j]
                if score > kpt_thr and visible:
                    x = int(max(min(t_points[i, j, 0], w - 1), 0))
                    y = int(max(min(t_points[i, j, 1], h - 1), 0))
                    valid[j] = 1
                    pz = pred_3d_instances.keypoints[i, j, 2] / 1000.0
                    # dz = depth[y, x] / 1000.0
                    dz = get_pixel_depth(depth, x, y) / 1000.0
                    if abs(dz - center_depth) < abs(pz):
                        z = dz
                    else:
                        z = center_depth - pz
                    temp_points.append([x, y, z])
            if np.sum(valid) == 17:
                # origin_points.append(temp_points)
                temp_np = np.array(temp_points)

                # Compute indices
                xx = temp_np[:, 0]
                yy = temp_np[:, 1]
                xx = (xx - CX) / FX
                yy = (yy - CY) / FY
                z_depth = temp_np[:, 2]
                pts_cam = np.stack((xx * z_depth, yy * z_depth, z_depth), axis=-1)
                # print("k_pts_cam", k_pts_cam.dtype)
                pix_ones = np.ones((temp_np.shape[0], 1))
                pts4 = np.concatenate((pts_cam, pix_ones), axis=1)
                # print("k_pts4", k_pts4.dtype)
                pts = (k_c2w @ pts4.T).T[:, :3]

                show_points.append(pts.tolist())
                depth_mean.append(np.mean(temp_np[:, 2]))
                depth_id.append(track_id[i])

                if i < len(bboxes_xywh): #注意后续要修改
                    b_x, b_y, b_w, b_h = bboxes_xywh[i]
                    key_distance = ((b_x - w/2)**2 + (b_y - h/2)**2) / ((w/2)**2 + (h/2)**2)
                    key_area = 1 - b_w * b_h / (w * h)
                    key_error = alpha * key_distance + (1 - alpha) * key_area
                    key_errors.append(key_error)
                    #key_points = pts.tolist()
                else:
                    key_errors.append(10000)

    if len(key_errors):
        min_error = min(key_errors)
        min_idx = key_errors.index(min_error)
        key_points = copy.deepcopy(show_points[min_idx])
        del show_points[min_idx]
        del depth_mean[min_idx]
        del depth_id[min_idx]
    else:
        key_points = []

    # crop_points = []
    # crop_mean = []
    # crop_id = []
    #
    # if len(key_errors):
    #     min_error = min(key_errors)
    #     min_idx = key_errors.index(min_error)
    #     key_points = show_points[min_idx]
    #     for i in range(len(key_errors)):
    #         if i != min_idx:
    #             crop_points.append(show_points[i])
    #             crop_mean.append(depth_mean[i])
    #             crop_id.append(depth_id[i])
    # else:
    #     crop_points = show_points
    #     crop_mean = de
    #     crop_id =
    #     key_points = []

    return np.array(show_points), np.array(key_points), depth_mean, depth_id


def init_camera(
    w=640,
    h=480,
    y_angle=0.0,
    center_dist=0.5,
    cam_height=0.0,
    cam_width=0.0,
    f_ratio=0.7,
):  # w=640, h=480, y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82
    ry = y_angle * np.pi / 180
    w2c = np.array(
        [
            [np.cos(ry), 0.0, -np.sin(ry), cam_width],
            [0.0, 1.0, 0.0, cam_height],
            [np.sin(ry), 0.0, np.cos(ry), center_dist],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def visualize(scene_path, cfg):
    # yolo_model = get_yolov5('ultralytics/yolov5', 'yolov5s')
    yolo_model = YOLO(
        "yolov8n.pt"
    )  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_device("234222304656")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    size = (640, 480)

    # Start streaming
    profile = pipeline.start(config)

    for i in range(10):
        pipeline.wait_for_frames()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    profile_d = profile.get_stream(rs.stream.depth)
    profile_c = profile.get_stream(rs.stream.color)
    intr_d = profile_d.as_video_stream_profile().get_intrinsics()
    intr_c = profile_c.as_video_stream_profile().get_intrinsics()
    print("深度传感器内参：", intr_d)
    print("RGB相机内参:", intr_c)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    temporal = rs.temporal_filter()

    # Load Scene Data
    init_w2c, init_k = load_camera(cfg, scene_path)

    scene_data, scene_depth_data, all_w2cs = load_scene_data(
        scene_path, init_w2c, init_k
    )

    # vis.create_window()
    vis = o3d.visualization.Visualizer()

    vis.create_window(
        width=int(cfg["viz_w"] * cfg["view_scale"]),
        height=int(cfg["viz_h"] * cfg["view_scale"]),
        visible=True,
    )

    (
        im,
        depth,
        sil,
    ) = render(init_w2c, init_k, scene_data, scene_depth_data, cfg)
    init_pts, init_cols = rgbd2pcd(im, depth, init_w2c, init_k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg["viz_w"]
    h = cfg["viz_h"]

    if cfg["visualize_cams"]:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap("cool")
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                w, h, init_k, all_w2cs[i_t], frustum_size
            )
            frustum.paint_uniform_color(
                np.array(cam_colormap(i_t * norm_factor / num_t)[:3])
            )
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap("cool")
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(
                np.array(
                    line_colormap(
                        (line_t * norm_factor / total_num_lines) + norm_factor
                    )[:3]
                )
            )
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # Initialize View Control
    view_k = init_k * cfg["view_scale"]
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg["offset_first_viz_cam"]:
        view_w2c = init_w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = init_w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg["viz_h"] * cfg["view_scale"])
    cparams.intrinsic.width = int(cfg["viz_w"] * cfg["view_scale"])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.background_color = [0, 0, 0]
    render_options.point_size = cfg["view_scale"]
    render_options.light_on = False

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parse_args()

    yolo_model = YOLO(
        "yolov8n.pt"
    )  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可

    pose_estimator = init_model(
        args.pose3d_estimator_config,
        args.pose3d_estimator_checkpoint,
        device=args.device.lower(),
    )

    det_kpt_color = pose_estimator.dataset_meta.get("keypoint_colors", None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get("skeleton_links", None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        "skeleton_link_colors", None
    )

    pose_estimator.cfg.model.test_cfg.mode = "vis"
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.det_dataset_link_color = (
        det_dataset_link_color  # noqa: E501
    )
    pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.kpt_color = det_kpt_color
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    pose_est_results_list = []
    pred_instances_list = []
    last_depth_mean = []
    pose_est_results = []
    last_depth_id = []
    persons = []
    frame_idx = 0
    next_id = 0
    get_start = 0
    start_time = time.time()
    num_timesteps = len(scene_data)
    first_view_w2c = view_w2c
    curr_timestep = 0
    angle_index = 0
    count = 0

    # Interactive Rendering
    while True:
        # passed_time = time.time() - start_time
        # passed_frames = passed_time * fps
        # t = int(passed_frames % num_timesteps)
        count += 1
        passed_time = time.time() - start_time
        passed_frames = passed_time * cfg["viz_fps"]
        curr_timestep = int(passed_frames % num_timesteps)
        # print("curr_timestep", curr_timestep)

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        aligned_depth_frame = temporal.process(aligned_depth_frame)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        depth_image = np.where(
            (depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH), depth_image, 0
        )

        # depth_image = cv2.medianBlur(depth_image, 3)

        if FORCE_LOOP:
            # num_loops = 1.4
            # y_angle = 360*t*num_loops / num_timesteps
            # print("y_angle", y_angle)
            w2c, i_k = init_camera(
                y_angle=angle_list[angle_index],
                cam_width=translate_list[angle_index],
                cam_height=0.0,
                center_dist=dist_list[angle_index],
            )
            angle_index += 1
            if angle_index >= len(angle_list):
                angle_index = 0
            # cam_params = view_control.convert_to_pinhole_camera_parameters()
            # cam_params.extrinsic = w2c
            # view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            # view_k = cam_params.intrinsic.intrinsic_matrix
            # k = view_k / cfg['view_scale']
            # print("k",k)
            # k[2, 2] = 1
            # Get Current View Camera
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            print("cam_w2c", cam_params.extrinsic)
            cam_params.extrinsic = w2c
            print("w2c",w2c)
            # w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
            view_control.convert_from_pinhole_camera_parameters(
                cam_params, allow_arbitrary=True
            )
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / cfg["view_scale"]
            k[2, 2] = 1
        else:  # Interactive control
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / cfg["view_scale"]
            k[2, 2] = 1
            w2c = cam_params.extrinsic
            print("w2c",w2c)

        color_image.flags.writeable = False

        pose_est_results_last = pose_est_results

        (pose_est_results, pose_est_results_list, pred_3d_instances, next_id, bboxes_xywh, plotted_image) = (
            process_one_image(
                args=args,
                detector=yolo_model,
                frame=color_image,
                frame_idx=frame_idx,
                pose_estimator=pose_estimator,
                pose_est_results_last=pose_est_results_last,
                pose_est_results_list=pose_est_results_list,
                next_id=next_id,
                visualize_frame=mmcv.bgr2rgb(color_image),
                visualizer=visualizer,
            )
        )

        # reverse changes
        color_image.flags.writeable = True
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        track_id = []
        for t_id in range(len(pose_est_results)):
            track_id.append(pose_est_results[t_id].track_id)

        show_points, key_points, depth_mean, depth_id = get_show(
            color_image, depth_image, pred_3d_instances, init_k, track_id, bboxes_xywh,
        )

        print("show_points", show_points)
        print("key_points", key_points)

        if len(show_points):
            # last_points = copy.deepcopy(show_points)
            # last_depth_mean = copy.deepcopy(depth_mean)
            # last_depth_id = copy.deepcopy(depth_id)
            # if len(persons):
            #     for person in persons:
            #         person.remove_geometry(vis)
            #         del person
            # persons = []
            # for i in range(show_points.shape[0]):
            #     person = PersonGeometry(show_points[i])
            #     person.add_geometry(vis)
            #     persons.append(person)
            if get_start:
                depth_error = 0
                depth_len = 0
                for now_id in range(len(depth_mean)):
                    for last_id in range(len(last_depth_mean)):
                        if abs(depth_id[now_id] - last_depth_id[last_id]) < 1e-5:
                            depth_error += abs(
                                depth_mean[now_id] - last_depth_mean[last_id]
                            )
                            depth_len += 1
                if depth_len:
                    mean_error = depth_error / depth_len
                else:
                    mean_error = 0
                if mean_error < 0.5:
                    #last_points = copy.deepcopy(show_points)
                    last_depth_mean = copy.deepcopy(depth_mean)
                    last_depth_id = copy.deepcopy(depth_id)
                    #last_key_points = copy.deepcopy(key_points)
                    if len(persons):
                        for person in persons:
                            person.remove_geometry(vis)
                            del person
                    persons = []
                    for i in range(show_points.shape[0]):
                        person = PersonGeometry(show_points[i])
                        person.add_geometry(vis)
                        persons.append(person)
                    if len(key_points):
                        person = PersonGeometry(key_points, True)
                        person.add_geometry(vis)
                        persons.append(person)
                    # print("mean_error", mean_error)
                else:
                    get_start = 0
            else:
                #last_points = copy.deepcopy(show_points)
                last_depth_mean = copy.deepcopy(depth_mean)
                last_depth_id = copy.deepcopy(depth_id)
                if len(persons):
                    for person in persons:
                        person.remove_geometry(vis)
                        del person
                persons = []
                for i in range(show_points.shape[0]):
                    person = PersonGeometry(show_points[i])
                    person.add_geometry(vis)
                    persons.append(person)
                if len(key_points):
                    person = PersonGeometry(key_points, True)
                    person.add_geometry(vis)
                    persons.append(person)
                get_start = 1

        if cfg["render_mode"] == "centers":
            pts = o3d.utility.Vector3dVector(
                scene_data["means3D"].contiguous().double().cpu().numpy()
            )
            cols = o3d.utility.Vector3dVector(
                scene_data["colors_precomp"].contiguous().double().cpu().numpy()
            )
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
            if cfg["show_sil"]:
                im = (1 - sil).repeat(3, 1, 1)
            # print("init_w2c", init_w2c)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

        cv2.imshow('detection', plotted_image)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        view_control.convert_from_pinhole_camera_parameters(
            cam_params, allow_arbitrary=True
        )

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":

    args = parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    # Visualize Final Reconstruction
    visualize(scene_path, viz_cfg)
