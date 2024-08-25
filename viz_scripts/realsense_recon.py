import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import mediapipe as mp
import pyrealsense2 as rs
import copy
import time

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from ycy_utils.common_utils import seed_everything
from ycy_utils.recon_helpers import setup_camera
from ycy_utils.slam_helpers import get_depth_and_silhouette
from ycy_utils.slam_external import build_rotation

from ultralytics import YOLO  # 将YOLOv8导入到该py文件中

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

MIN_DEPTH = 200  # 20mm
MAX_DEPTH = 5000  # 10000mm

FORCE_LOOP = True

alpha = 0.6

MAX_ANGLE = 10
NUM_ANGLE = 101
MAX_TRANSLATE = 0.2
MAX_DIST = 0.1

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

n_l = 2
n_c = 2*n_l+1
num = n_c**3

pose_keypoints_id = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

#pose_link = np.array([[0, 1], [1, 7], [7, 6], [6, 0], [1, 3], [3, 5], [0, 2], [2, 4], [6, 8], [8, 10], [7, 9], [9, 11]])

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
init_points = np.ones((12*num, 2)).tolist()

# init_depth = [1.6240, 1.6190, 1.0450, 1.0440, 1.0530, 0.9610, 1.0700, 1.1000, 1.8490,
#         1.8900, 1.4620, 1.2900]
init_depth = np.ones(12*num).tolist()

k_c2w = torch.tensor(  [[ 0.99923, -0.01485,  0.03636, -0.435],
                        [ 0.02495,  0.95499, -0.29558,  0.37541],
                        [-0.03034,  0.29626,  0.95463,  0.43624],
                        [ 0.,       0.,       0.,       1.     ]] , device="cuda:0", dtype=torch.float32)

def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k

def polygon():
    # 绘制顶点
    polygon_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 5]])
    lines = [[0, 1], [1, 2], [2, 3]]  # 连接的顺序，封闭链接[3, 0]
    color = [[1, 0, 0] for i in range(len(lines))]
    # 添加顶点，点云
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0, 1])  # 点云颜色

    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    return lines_pcd, points_pcd
def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    # Check if Gaussians are Isotropic or Anisotropic
    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], first_frame_w2c),
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    return rendervar, depth_rendervar, all_w2cs

def pose_link(num_people):
    pose_links = []
    for j in range(num_people):
        n_p = 12 * num * j
        for i in range(num):
            pose_links.extend(
                [[i + 0 * num + n_p, i + 1 * num + n_p], [i + 1 * num + n_p, i + 7 * num + n_p],
                 [i + 7 * num + n_p, i + 6 * num + n_p],
                 [i + 6 * num + n_p, i + 0 * num + n_p], [i + 1 * num + n_p, i + 3 * num + n_p],
                 [i + 3 * num + n_p, i + 5 * num + n_p],
                 [i + 0 * num + n_p, i + 2 * num + n_p], [i + 2 * num + n_p, i + 4 * num + n_p],
                 [i + 6 * num + n_p, i + 8 * num + n_p],
                 [i + 8 * num + n_p, i + 10 * num + n_p], [i + 7 * num + n_p, i + 9 * num + n_p],
                 [i + 9 * num + n_p, i + 11 * num + n_p]])
    #pose_links = np.array(use_lines)
    return pose_links

# def findPose(img, pose, draw=True):
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换成灰度图片
#     results = pose.process(imgRGB)  # 处理 RGB 图像并返回检测到的最突出人物的姿势特征点。
#     if results.pose_landmarks:
#         if draw:
#             mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose .POSE_CONNECTIONS)
#             # results.pose_landmarks画点 mpPose.POSE_CONNECTIONS连线
#     return img

def findPosition(img, depth, results, k, draw=True):
    show_points = []
    show_depth = []
    sucess = 1
    h, w, c = img.shape  # 返回图片的(高,宽,位深)
    if results.pose_landmarks:
        l_hip_x = int(w * results.pose_landmarks.landmark[23].x)
        l_hip_y = int(h * results.pose_landmarks.landmark[23].y)
        r_hip_x = int(w * results.pose_landmarks.landmark[24].x)
        r_hip_y = int(h * results.pose_landmarks.landmark[24].y)
        if l_hip_x > 0 and l_hip_y > 0 and r_hip_x > 0 and r_hip_y > 0 and l_hip_x < w and l_hip_y < h and r_hip_x < w and r_hip_y < h:
            if depth[l_hip_y, l_hip_x] >= MIN_DEPTH and depth[r_hip_y, r_hip_x] >= MIN_DEPTH :
                center_depth = (depth[int(l_hip_y), int(l_hip_x)] + depth[int(r_hip_y), int(r_hip_x)]) / 2000.0
                cd = 2 * center_depth / (k[0, 0] + k[1, 1])
                for id, lm in enumerate(results.pose_landmarks.landmark):  # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                    if id in pose_keypoints_id:  # only save keypoints that are indicated in pose_keypoints
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), results.pose_world_landmarks.landmark[id].z + center_depth  # lm.x  lm.y是比例  乘上总长度就是像素点位置
                        cx = max(min(cx,w-n_l-1),n_l)
                        cy = max(min(cy,h-n_l-1),n_l)
                        dz = depth[cy, cx] / 1000.0
                        diff = abs(dz - center_depth)

                        #rz = dz if diff < abs(lm.z) else cz
                        rz =cz

                        if rz > 1e-2:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # 画蓝色圆圈
                            for m in range(-n_l, n_l + 1):
                                for n in range(-n_l, n_l + 1):
                                    for d in range(-n_l, n_l + 1):
                                        show_points.append([cx + m, cy + n])
                                        show_depth.append(rz + cd * d)
                        else:
                            sucess = 0
                            break
            else:
                sucess = 0
        else:
            sucess = 0
    else:
        sucess = 0

    return show_points, show_depth, sucess

def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
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
            prefiltered=cam.prefiltered
        )
        im, _, depth, _, _,= Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
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
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def ycy_rgbd2pcd(color, depth, w2c, intrinsics, cfg, s_points, s_depth, k_c2w, use_lines): #lines
    width, height = color.shape[2], color.shape[1]
    lines = np.array(use_lines)
    # key_lines = np.array(k_lines)
    # print("pose_link", lines.shape)
    # lines = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9],
    #          [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]])  # 连接的顺序，封闭链接

    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    s_xx = s_points[:,0]
    # k_xx = k_points[:, 0]

    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    s_yy = s_points[:,1]
    # k_yy = k_points[:, 1]

    xx = (xx - CX) / FX
    s_xx = (s_xx - CX) / FX
    # k_xx = (k_xx - CX) / FX

    yy = (yy - CY) / FY
    s_yy = (s_yy - CY) / FY
    # k_yy = (k_yy - CY) / FY

    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    #print((k_xx * k_depth).shape)

    #print(k_yy.shape)

    s_pts_cam = torch.stack((s_xx * s_depth, s_yy * s_depth, s_depth), dim=-1)
    #print("k_pts_cam", k_pts_cam.dtype)
    s_pix_ones = torch.ones(s_points.shape[0], 1).cuda().float()
    s_pts4 = torch.cat((s_pts_cam, s_pix_ones), dim=1)
    #print("k_pts4", k_pts4.dtype)
    s_pts = (k_c2w @ s_pts4.T).T[:, :3]

    # k_pts_cam = torch.stack((k_xx * k_depth, k_yy * k_depth, k_depth), dim=-1)
    # #print("k_pts_cam", k_pts_cam.dtype)
    # k_pix_ones = torch.ones(k_points.shape[0], 1).cuda().float()
    # k_pts4 = torch.cat((k_pts_cam, k_pix_ones), dim=1)
    # #print("k_pts4", k_pts4.dtype)
    # k_pts = (k_c2w @ k_pts4.T).T[:, :3]

    #print("s_pts", s_pts)

    # 绘制颜色
    green_color = torch.tensor([0.0, 1.0, 0.0], device="cuda:0")
    red_color = torch.tensor([1.0, 0.0, 0.0], device="cuda:0")
    blue_color = torch.tensor([0.0, 0.0, 1.0], device="cuda:0")
    yellow_color = torch.tensor([1.0, 1.0, 0.0], device="cuda:0")

    spoints_cols = green_color.repeat(s_points.shape[0], 1)
    slines_cols = blue_color.repeat(lines.shape[0], 1)

    # kpoints_cols = red_color.repeat(k_points.shape[0], 1)
    #klines_cols = green_color.repeat(key_lines.shape[0], 1)

    # 绘制线条
    # lines_pcd = o3d.geometry.LineSet()
    pose_lineset = o3d.utility.Vector2iVector(np.ascontiguousarray(lines, np.int32))
    pose_colors = o3d.utility.Vector3dVector(slines_cols.contiguous().double().cpu().numpy())  # 线条颜色
    pose_points = o3d.utility.Vector3dVector(s_pts.contiguous().double().cpu().numpy())

    pts = torch.cat((pts, s_pts), 0)

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
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
        yolo_model = yolo_model.to('cuda')
    else:
        print("GPU device not found. Using CPU instead.")

    # since we are only interested in detecting a person
    yolo_model.classes = [0]

    return yolo_model

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
    yolo_model = YOLO("yolov8n.pt")  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可

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
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

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
    print("Depth Scale is: " , depth_scale)

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

    # 创建用于保存视频的 VideoWriter 对象
    # out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

    temporal = rs.temporal_filter()

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load Scene Data
    init_w2c, init_k = load_camera(cfg, scene_path)

    scene_data, scene_depth_data, all_w2cs = load_scene_data(scene_path, init_w2c, init_k)

    # vis.create_window()
    vis = o3d.visualization.Visualizer()

    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, sil = render(init_w2c, init_k, scene_data, scene_depth_data, cfg)

    init_pose_links = pose_link(1)

    last_points = copy.deepcopy(init_points)
    last_depth = copy.deepcopy(init_depth)
    last_pose_links = init_pose_links

    t_init_points = torch.tensor(init_points, device="cuda:0", dtype=torch.float32)
    t_init_depth = torch.tensor(init_depth, device="cuda:0", dtype=torch.float32)

    init_pts, init_cols, init_pose_lineset, init_pose_colors, init_pose_points = ycy_rgbd2pcd(im, depth, init_w2c, init_k, cfg,
                                                                                              t_init_points, t_init_depth, k_c2w, init_pose_links)
    #init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols

    vis.add_geometry(pcd)

    pose_lines = o3d.geometry.LineSet()

    pose_lines.lines = init_pose_lineset
    pose_lines.colors = init_pose_colors
    pose_lines.points = init_pose_points

    vis.add_geometry(pose_lines)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, init_k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines) + norm_factor)[:3]))
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
    view_k = init_k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = init_w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = init_w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.background_color = [0, 0, 0]
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    angle_index = 0

    # Interactive Rendering
    while True:
        start = time.time()
        # sucess = 1
        # cam_params = view_control.convert_to_pinhole_camera_parameters()
        # view_k = cam_params.intrinsic.intrinsic_matrix
        # k = view_k / cfg['view_scale']
        # k[2, 2] = 1
        # w2c = cam_params.extrinsic

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

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        aligned_depth_frame = temporal.process(aligned_depth_frame)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        depth_image = np.where((depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH), depth_image, 0)

        color_image = np.asanyarray(color_frame.get_data())

        # the BGR image to RGB.
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        color_image.flags.writeable = False

        # results = pose.process(color_image)
        # yolo_results = yolo_model(color_image)
        # yolo_results = yolo_model.predict(source=color_image, device='cuda:0', classes=0)
        yolo_results = yolo_model.track(color_image, persist=True, classes=0, device=0)
        print(yolo_results[0].orig_shape[1])
        annotated_frame = yolo_results[0].plot()
        #yolo_results = yolo_model(color_image, stream=True)

        # reverse changes
        color_image.flags.writeable = True
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # This array will contain crops of images in case we need it
        img_list = []
        show_points = []
        show_depth = []

        # we need some extra margin bounding box for human crops to be properly detected
        MARGIN = 5
        num_people = 0  # 重置人数为0
        min_error = 10000

        for (xmin, ymin, xmax, ymax) in yolo_results[0].boxes.xyxy.tolist():
        #for (xmin, ymin, xmax, ymax, confidence, clas) in yolo_results.xyxy[0].tolist():
            # Media pose prediction
            cropped_image = color_image[max(int(ymin) - MARGIN, 0):min(int(ymax) + MARGIN, size[1]), min(int(xmin) - MARGIN, 0):max(int(xmax) + MARGIN, size[0])]
            cropped_depth_image = depth_image[max(int(ymin) - MARGIN, 0):min(int(ymax) + MARGIN, size[1]), min(int(xmin) - MARGIN, 0):max(int(xmax) + MARGIN, size[0])]
            #cropped_image = findPose(cropped_image)
            imgRGB = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)  # 将BGR格式转换成灰度图片
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    cropped_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            #lmLists.append(detector.findPosition(cropped_image, depth_image))
            person_points, person_depth, sucess = findPosition(cropped_image, cropped_depth_image, results, init_k)
            if sucess == 0:
                continue

            # key_distance = (((xmin + xmax - w) / 2) ** 2 + ((ymin + ymax - h) / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
            # key_area = 1 - (xmax - xmin) * (ymax - ymin) / (w * h)
            # key_error = alpha * key_distance + (1 - alpha) * key_area
            # if key_error < min_error:
            #     key_points = person_points
            #     key_depth = person_depth

            show_points.extend(person_points)
            show_depth.extend(person_depth)
            img_list.append(cropped_image)

            num_people += 1  # 每检测到一个人，人数加1

        # # Draw the pose annotation on the image.
        # # Flip the image horizontally for a selfie-view display.
        # cv2.namedWindow('MediaPipe Pose', 0)
        # cv2.resizeWindow('MediaPipe Pose', 640, 480)
        # cv2.imshow('MediaPipe Pose', cv2.flip(color_image, 1))
        # cv2.waitKey(5)
        # #cv2.destroyAllWindows()


        # # check for keypoints detection
        # show_points = []
        # show_depth = []
        # if results.pose_landmarks == None:
        #     # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
        #     sucess = 0
        # else:
        #     for i, landmark in enumerate(results.pose_landmarks.landmark):
        #         if i not in pose_keypoints_id: continue  # only save keypoints that are indicated in pose_keypoints
        #         pxl_x = landmark.x * color_image.shape[1]
        #         pxl_y = landmark.y * color_image.shape[0]
        #         pxl_x = int(round(pxl_x))
        #         pxl_y = int(round(pxl_y))
        #         if pxl_x < n_l or pxl_x >= (color_image.shape[1]-n_l-1) or pxl_y < n_l or pxl_y >= (color_image.shape[0]-n_l-1):
        #             sucess = 0
        #             break
        #         # if depth_image[pxl_y][pxl_y] < 1e-5:
        #         #     sucess = 0
        #         #     break
        #         #cv2.circle(color_image, (pxl_x, pxl_y), 3, (0, 0, 255), -1)  # add keypoint detection points into figure
        #         all_depth = []
        #         for m in range(-n_l,n_l+1):
        #             for n in range(-n_l,n_l+1):
        #                 show_points.append([pxl_x+m, pxl_y+n])
        #                 all_depth.append(depth_image[pxl_y+n][pxl_y+m] / 1000.0)
        #         all_depth.sort()
        #         d_len = len(all_depth)
        #         d_m = d_len // 2
        #         if all_depth[d_m] < 1e-5:
        #             sucess = 0
        #             break
        #         current_depth = np.ones_like(all_depth) * all_depth[d_m]
        #         #print("current", current_depth.tolist())
        #         show_depth.extend(current_depth.tolist())

        # we need some extra margin bounding box for human crops to be properly detected

        if num_people > 0: #update
            last_points = []
            last_depth = []
            last_pose_links = []
            # last_key_points = []
            # last_key_depth = []
            # last_key_pose_links = []
            last_points = copy.deepcopy(show_points)
            last_depth = copy.deepcopy(show_depth)
            pose_links = pose_link(num_people)
            last_pose_links = copy.deepcopy(pose_links)

            # last_key_points = copy.deepcopy(key_points)
            # last_key_depth = copy.deepcopy(key_depth)
            # key_pose_links = pose_link(1)
            # last_key_pose_links = copy.deepcopy(key_pose_links)

        t_show_depth = torch.tensor(last_depth, device="cuda:0", dtype=torch.float32)
        t_show_points = torch.tensor(last_points, device="cuda:0", dtype=torch.float32)
        # t_key_depth = torch.tensor(last_key_depth, device="cuda:0", dtype=torch.float32)
        # t_key_points = torch.tensor(last_key_points, device="cuda:0", dtype=torch.float32)

        # cv2.imshow('pose_estimation', color_image)
        # key = cv2.waitKey(1)
        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break

        print("depth",t_show_depth.size())
        print("points",t_show_points.size())

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1 - sil).repeat(3, 1, 1)
            # if continue_flag == 1:
            #     pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
            pts, cols, pose_lineset, pose_colors, pose_points = ycy_rgbd2pcd(im, depth, w2c, k, cfg,
                                                                             t_show_points, t_show_depth, k_c2w, last_pose_links)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        pose_lines.lines = pose_lineset
        pose_lines.colors = pose_colors
        pose_lines.points = pose_points
        vis.update_geometry(pose_lines)

        # mp_drawing.draw_landmarks(
        #     color_image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #

        # # Display the resulting image with person count
        # cv2.putText(color_image, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('Activity Recognition', color_image)
        # # Write the frame to video file
        # #out.write(color_image)

        end = time.time()
        fps = 1 / (end - start)
        fps = "%.2f fps" % fps
        # 实时显示帧数
        #annotated_frame = cv2.flip(annotated_frame, 1)
        cv2.putText(annotated_frame, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "FPS {0}".format(fps), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Pose', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    pipeline.stop()
    #out.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

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