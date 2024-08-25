from argparse import ArgumentParser
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation

from pyorbbecsdk import *
from orbbec_utils import frame_to_bgr_image

import logging
import mimetypes
import time

import json_tricks as json

import mmcv
import mmengine
from mmengine.logging import print_log

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import split_instances

ESC_KEY = 27
MIN_DEPTH = 200  # 20mm
MAX_DEPTH = 5000  # 10000mm


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
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth, _, _, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
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

def ycy_rgbd2pcd(color, depth, w2c, intrinsics, cfg, s_points, s_depth, k_c2w, lines):
    width, height = color.shape[2], color.shape[1]
    # lines = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9],
    #          [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]])  # 连接的顺序，封闭链接

    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    s_xx = s_points[:,0]
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    s_yy = s_points[:,1]
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

    #print((k_xx * k_depth).shape)

    #print(k_yy.shape)

    s_pts_cam = torch.stack((s_xx * s_depth, s_yy * s_depth, s_depth), dim=-1)
    #print("k_pts_cam", k_pts_cam.dtype)
    s_pix_ones = torch.ones(s_points.shape[0], 1).cuda().float()
    s_pts4 = torch.cat((s_pts_cam, s_pix_ones), dim=1)
    #print("k_pts4", k_pts4.dtype)
    s_pts = (k_c2w @ s_pts4.T).T[:, :3]

    print("s_pts", s_pts)

    # 绘制颜色
    green_color = torch.tensor([0.0, 1.0, 0.0], device="cuda:0")
    spoints_cols = green_color.repeat(s_points.shape[0], 1)
    red_color = torch.tensor([1.0, 0.0, 0.0], device="cuda:0")
    klines_cols = red_color.repeat(lines.shape[0], 1)

    # 绘制线条
    # lines_pcd = o3d.geometry.LineSet()
    pose_lineset = o3d.utility.Vector2iVector(np.ascontiguousarray(lines, np.int32))
    pose_colors = o3d.utility.Vector3dVector(klines_cols.contiguous().double().cpu().numpy())  # 线条颜色
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

def polygon():
    # 绘制顶点
    polygon_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 5]])
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]  # 连接的顺序，封闭链接
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

def process_one_image(args,
                      img,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # inference a single image
    batch_results = inference_bottomup(pose_estimator, img)
    results = batch_results[0]

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=False,
            draw_heatmap=args.draw_heatmap,
            show_kpt_idx=args.show_kpt_idx,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    return results.pred_instances

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')

    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')

    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')

    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')

    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')

    parser.add_argument("-m", "--mode",
                        help="align mode, HW=hardware mode,SW=software mode,NONE=disable align",
                        type=str, default='HW')

    parser.add_argument("-s", "--enable_sync", help="enable sync", type=bool, default=True)

    args = parser.parse_args()

    return args

def visualize(scene_path, cfg, args):

    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # build visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()

    align_mode = args.mode
    enable_sync = args.enable_sync
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        #print("\nOBSensorType.COLOR_SENSOR", OBSensorType.COLOR_SENSOR)
        #color_profile = profile_list.get_default_video_stream_profile()
        color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        #print("\nOBSensorType.DEPTH_SENSOR", OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        #depth_profile = profile_list.get_default_video_stream_profile()
        depth_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y12, 30)
        assert depth_profile is not None
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
        print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return
    if align_mode == 'HW':
          if device_pid == 0x066B:
            #Femto Mega does not support hardware D2C, and it is changed to software D2C
             config.set_align_mode(OBAlignMode.SW_MODE)
          else:
             config.set_align_mode(OBAlignMode.HW_MODE)
    elif align_mode == 'SW':
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.DISABLE)
    if enable_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(e)
    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return


    # Load Scene Data
    w2c, k = load_camera(cfg, scene_path)

    scene_data, scene_depth_data, all_w2cs = load_scene_data(scene_path, w2c, k)

    # vis.create_window()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    init_lineset = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]])
    #init_lineset = np.array([[5,11]])

    #init_depth = torch.randint(500, 600, (17,), device="cuda:0", dtype=torch.float32)

    #init_points = torch.randint(300, 400, (17,2), device="cuda:0", dtype=torch.float32)

    init_depth = torch.ones((17,), device="cuda:0", dtype=torch.float32)

    init_points = 100 * torch.ones((17,2), device="cuda:0", dtype=torch.float32)

    # init_c2w = torch.tensor([[1.0, 0.0, 0.0, 0.0],
    #                       [0.0, 1.0, 0.0, 0.0],
    #                       [0.0, 0.0, 1.0, 0.0],
    #                       [0.0, 0.0, 0.0, 1.0]], device="cuda:0", dtype=torch.float32)

    #init_c2w = torch.inverse(torch.tensor(w2c).cuda().float())

    k_c2w = torch.tensor([[0.967466295,  0.218611643,  0.127350003,  1.49842167],
                          [-0.221743584,  0.975045383,  0.0107821785,  0.421921223],
                          [-0.121814981, -0.038670443,  0.991799235, -0.598875165],
                          [0.0, 0.0, 0.0, 1.0]], device="cuda:0", dtype=torch.float32)

    im, depth, sil, = render(w2c, k, scene_data, scene_depth_data, cfg)
    init_pts, init_cols, init_pose_lineset, init_pose_colors, init_pose_points = ycy_rgbd2pcd(im, depth, w2c, k, cfg, init_points, init_depth, k_c2w, init_lineset)
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
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
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
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # num = 1 #81
    #
    # use_lines = []
    #
    # for i in range(num):
    #     use_lines.extend([[i + 0 * num, i + 1 * num], [i + 0 * num, i + 2 * num], [i + 1 * num, i + 2 * num],
    #                       [i + 1 * num, i + 3 * num], [i + 2 * num, i + 4 * num],
    #                       [i + 3 * num, i + 5 * num], [i + 4 * num, i + 6 * num], [i + 5 * num, i + 6 * num],
    #                       [i + 5 * num, i + 7 * num], [i + 5 * num, i + 11 * num],
    #                       [i + 6 * num, i + 8 * num], [i + 6 * num, i + 12 * num], [i + 7 * num, i + 9 * num],
    #                       [i + 8 * num, i + 10 * num], [i + 11 * num, i + 12 * num],
    #                       [i + 11 * num, i + 13 * num], [i + 12 * num, i + 14 * num], [i + 13 * num, i + 15 * num],
    #                       [i + 14 * num, i + 16 * num]])
    #
    # np_lines = np.array(use_lines)

    np_lines = init_lineset

    # Interactive Rendering
    while True:
        continue_flag = 0
        frames: FrameSet = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        # covert to RGB format
        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("failed to convert frame to image")
            continue
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)

        depth_data = depth_data * 5
        depth_data = depth_data.astype(np.uint16)
        m_depth = depth_data / 5000.0

        #print("depth.shape",depth_data.shape)

        pred_instances = process_one_image(args, color_image, model, visualizer,
                                           0.001)

        k_points = pred_instances.get('transformed_keypoints',
                                      pred_instances.keypoints)
        #print(k_points)

        show_points = []
        show_depth = []

        #for k in range(k_points.shape[0]):

        for l in range(k_points.shape[1]):

            k_x = int(k_points[0][l][0])
            k_y = int(k_points[0][l][1])

            # if k_x < 4:
            #     k_x = 4
            # if k_x > 635:
            #     k_x = 635
            # if k_y < 4:
            #     k_y = 4
            # if k_y > 475:
            #     k_y = 475

            if k_x < 0 or k_x > 639 or k_y < 0 or k_y > 479:
                continue_flag = 1
                break

            #show_k_points.append([k_x, k_y])

            #k_depth.append(m_depth[k_y][k_x])

            # for i in range(-4, 5):
            #     for j in range(-4, 5):
            #         show_points.append([k_x+i, k_y+j])
            #         show_depth.append(m_depth[k_y][k_x])


            show_points.append([k_x, k_y])
            show_depth.append(m_depth[k_y][k_x])

        if continue_flag == 1:
            continue

        show_depth = torch.tensor(show_depth, device="cuda:0", dtype=torch.float32)

        #print("k_depth", k_depth.dtype)

        show_points = torch.tensor(show_points, device="cuda:0", dtype=torch.float32)

        #print("show_points", show_points.dtype)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1 - sil).repeat(3, 1, 1)
            pts, cols, pose_lineset, pose_colors, pose_points = ycy_rgbd2pcd(im, depth, w2c, k, cfg, show_points, show_depth, k_c2w, np_lines)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        pose_lines.lines = pose_lineset
        pose_lines.colors = pose_colors
        pose_lines.points = pose_points
        vis.update_geometry(pose_lines)

        # p_lines, p_points = polygon()
        # vis.update_geometry(p_lines)
        # vis.update_geometry(p_points)

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
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
    visualize(scene_path, viz_cfg, args)
