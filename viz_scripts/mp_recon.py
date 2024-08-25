import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import cv2
import torch
import torch.nn.functional as F
import torch.multiprocessing
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import mediapipe
from mediapipe.python.solutions import pose as mp_pose
import pyrealsense2 as rs
import copy
import time

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from ycy_utils.common_utils import seed_everything
from ycy_utils.recon_helpers import setup_camera
from ycy_utils.slam_helpers import get_depth_and_silhouette
from ycy_utils.slam_external import build_rotation

MIN_DEPTH = 200  # 20mm
MAX_DEPTH = 5000  # 10000mm

n_l = 2
n_c = 2*n_l+1
num = n_c**3

pose_keypoints_id = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

k_c2w = torch.tensor( [[ 0.95224, -0.26692, -0.14829, -0.30835],
                       [ 0.20802,  0.92259, -0.32489,  0.58543],
                       [ 0.22353,  0.27853,  0.93405,  0.07561],
                       [ 0.0    ,  0.0    ,  0.0    ,  1.0    ]] , device="cuda:0", dtype=torch.float32)


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
        im, _, depth, _, _,= Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil

def ycy_rgbd2pcd(color, depth, w2c, intrinsics, cfg, s_points, s_depth, k_c2w, lines): #lines
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

    print("depth",s_depth.shape)
    print("s_xx",s_xx.shape)
    print("s_yy", s_yy.shape)

    s_pts_cam = torch.stack((s_xx * s_depth, s_yy * s_depth, s_depth), dim=-1)
    s_pix_ones = torch.ones(s_points.shape[0], 1).cuda().float()
    s_pts4 = torch.cat((s_pts_cam, s_pix_ones), dim=1)
    s_pts = (k_c2w @ s_pts4.T).T[:, :3]

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

class PoseDetector():
    def __init__(self, scene_path, cfg, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.scene_path = scene_path
        self.cfg = cfg
        # Load Scene Data
        _, self.k = load_camera(cfg, scene_path)

        self.show_points = torch.ones((12*num,2),dtype=torch.float32)
        self.show_depth = torch.ones(12*num,dtype=torch.float32)

        self.mpDraw = mediapipe.solutions.drawing_utils
        self.mpPose = mediapipe.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, False, True,  # 这里的False 和True为默认
                                     self.detectionCon,
                                     self.trackCon)  # pose对象 1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、是否分割，5、减少抖动，6、检测阈值，7、跟踪阈值
        '''
        STATIC_IMAGE_MODE：如果设置为 false，该解决方案会将输入图像视为视频流。它将尝试在第一张图像中检测最突出的人，并在成功检测后进一步定位姿势地标。在随后的图像中，它只是简单地跟踪那些地标，而不会调用另一个检测，直到它失去跟踪，以减少计算和延迟。如果设置为 true，则人员检测会运行每个输入图像，非常适合处理一批静态的、可能不相关的图像。默认为false。
        MODEL_COMPLEXITY：姿势地标模型的复杂度：0、1 或 2。地标准确度和推理延迟通常随着模型复杂度的增加而增加。默认为 1。
        SMOOTH_LANDMARKS：如果设置为true，解决方案过滤不同的输入图像上的姿势地标以减少抖动，但如果static_image_mode也设置为true则忽略。默认为true。
        UPPER_BODY_ONLY：是要追踪33个地标的全部姿势地标还是只有25个上半身的姿势地标。
        ENABLE_SEGMENTATION：如果设置为 true，除了姿势地标之外，该解决方案还会生成分割掩码。默认为false。
        SMOOTH_SEGMENTATION：如果设置为true，解决方案过滤不同的输入图像上的分割掩码以减少抖动，但如果 enable_segmentation设置为false或者static_image_mode设置为true则忽略。默认为true。
        MIN_DETECTION_CONFIDENCE：来自人员检测模型的最小置信值 ([0.0, 1.0])，用于将检测视为成功。默认为 0.5。
        MIN_TRACKING_CONFIDENCE：来自地标跟踪模型的最小置信值 ([0.0, 1.0])，用于将被视为成功跟踪的姿势地标，否则将在下一个输入图像上自动调用人物检测。将其设置为更高的值可以提高解决方案的稳健性，但代价是更高的延迟。如果 static_image_mode 为 true，则忽略，人员检测在每个图像上运行。默认为 0.5。
        '''

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换成灰度图片
        self.results = self.pose.process(imgRGB)  # 处理 RGB 图像并返回检测到的最突出人物的姿势特征点。
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # results.pose_landmarks画点 mpPose.POSE_CONNECTIONS连线
        return img

    def findPosition(self, img, depth, draw=True):
        # print(results.pose_landmarks)
        #lmList = []
        show_points = []
        show_depth = []
        h, w, c = img.shape  # 返回图片的(高,宽,位深)
        sucess = 1
        if self.results.pose_landmarks:
            l_hip_x = int(w * self.results.pose_landmarks.landmark[23].x)
            l_hip_y = int(h * self.results.pose_landmarks.landmark[23].y)
            r_hip_x = int(w * self.results.pose_landmarks.landmark[24].x)
            r_hip_y = int(h * self.results.pose_landmarks.landmark[24].y)
            if l_hip_x > 0 and l_hip_y > 0 and r_hip_x > 0 and r_hip_y > 0 and l_hip_x < w and l_hip_y < h and r_hip_x < w and r_hip_y < h:
                if depth[l_hip_y, l_hip_x] > 1e-5 and depth[r_hip_y, r_hip_x] > 1e-5 :
                    center_depth = (depth[int(l_hip_y), int(l_hip_x)] + depth[int(r_hip_y), int(r_hip_x)]) / 2000.0
                    cd = 2 * center_depth / (self.k[0, 0] + self.k[1, 1])
                    for id, lm in enumerate(self.results.pose_landmarks.landmark):  # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z + center_depth)  # lm.x  lm.y是比例  乘上总长度就是像素点位置
                        cx = max(min(cx,w-n_l-1),n_l)
                        cy = max(min(cy,h-n_l-1),n_l)
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # 画蓝色圆圈
                        if id in pose_keypoints_id:  # only save keypoints that are indicated in pose_keypoints
                            for m in range(-n_l, n_l + 1):
                                for n in range(-n_l, n_l + 1):
                                    for d in range(-n_l, n_l + 1):
                                        show_points.append([cx + m, cy + n])
                                        show_depth.append(cz+cd*d)
                else:
                    sucess = 0
            else:
                sucess = 0
        else:
            sucess = 0
        return show_points, show_depth, sucess

    def pose_link(self, num_people):
        use_lines = []
        for j in range(num_people):
            n_p = 12 * num * j
            for i in range(num):
                use_lines.extend(
                    [[i + 0 * num + n_p, i + 1 * num + n_p], [i + 1 * num + n_p, i + 7 * num + n_p],
                     [i + 7 * num + n_p, i + 6 * num + n_p],
                     [i + 6 * num + n_p, i + 0 * num + n_p], [i + 1 * num + n_p, i + 3 * num + n_p],
                     [i + 3 * num + n_p, i + 5 * num + n_p],
                     [i + 0 * num + n_p, i + 2 * num + n_p], [i + 2 * num + n_p, i + 4 * num + n_p],
                     [i + 6 * num + n_p, i + 8 * num + n_p],
                     [i + 8 * num + n_p, i + 10 * num + n_p], [i + 7 * num + n_p, i + 9 * num + n_p],
                     [i + 9 * num + n_p, i + 11 * num + n_p]])
        pose_links = np.array(use_lines)
        return pose_links
    def run(self):

        # Load Scene Data
        w2c, k = load_camera(self.cfg, self.scene_path)

        scene_data, scene_depth_data, all_w2cs = load_scene_data(self.scene_path, w2c, k)

        # vis.create_window()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(self.cfg['viz_w'] * self.cfg['view_scale']),
                          height=int(self.cfg['viz_h'] * self.cfg['view_scale']),
                          visible=True)

        im, depth, sil = render(w2c, k, scene_data, scene_depth_data, self.cfg)

        t_init_points = self.show_points.clone()
        t_init_points = t_init_points.cuda(0)
        t_init_depth = self.show_depth.clone()
        t_init_depth = t_init_depth.cuda(0)

        init_pose_links = self.pose_link(1)

        init_pts, init_cols, init_pose_lineset, init_pose_colors, init_pose_points = ycy_rgbd2pcd(im, depth, w2c, k,
                                                                                                  self.cfg, t_init_points,
                                                                                                  t_init_depth, k_c2w,
                                                                                                  init_pose_links)
        pcd = o3d.geometry.PointCloud()
        pcd.points = init_pts
        pcd.colors = init_cols
        vis.add_geometry(pcd)

        pose_lines = o3d.geometry.LineSet()

        pose_lines.lines = init_pose_lineset
        pose_lines.colors = init_pose_colors
        pose_lines.points = init_pose_points
        vis.add_geometry(pose_lines)

        w = self.cfg['viz_w']
        h = self.cfg['viz_h']

        if self.cfg['visualize_cams']:
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
        view_k = k * self.cfg['view_scale']
        view_k[2, 2] = 1
        view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        if self.cfg['offset_first_viz_cam']:
            view_w2c = w2c
            view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
        else:
            view_w2c = w2c
        cparams.extrinsic = view_w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = int(self.cfg['viz_h'] * self.cfg['view_scale'])
        cparams.intrinsic.width = int(self.cfg['viz_w'] * self.cfg['view_scale'])
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

        render_options = vis.get_render_option()
        render_options.point_size = self.cfg['view_scale']
        render_options.light_on = False

        # Model
        # 加载模型文件
        # yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # 加载轻量级的模型文件
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # 检查是否有可用的GPU设备
        if torch.cuda.is_available():
            # 将yolo_model加载到GPU设备上
            yolo_model = yolo_model.to('cuda')
        else:
            print("GPU device not found. Using CPU instead.")

        # since we are only interested in detecting a person
        yolo_model.classes = [0]

        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

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
        # 获取摄像头的尺寸
        size = (640, 480)

        # Start streaming
        profile = pipeline.start(config)

        # 【Skip 5 first frames to give the Auto-Exposure time to adjust 跳过前5帧以设置自动曝光时间】
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

        # 创建用于保存视频的 VideoWriter 对象
        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

        num_people = 0  # 用于记录检测到的人数

        temporal = rs.temporal_filter()

        # Interactive Rendering
        while True:
            # start = time.time()
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / self.cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            # Validate that both frames are valid
            if not color_frame or not aligned_depth_frame:
                continue

            aligned_depth_frame = temporal.process(aligned_depth_frame)
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            depth_image = np.where((depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH), depth_image, 0)

            color_image = np.asanyarray(color_frame.get_data())

            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # making image writeable to False improves prediction
            image.flags.writeable = False

            result = yolo_model(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # This array will contain crops of images in case we need it
            img_list = []
            use_lines = []
            show_points = []
            show_depth = []

            # we need some extra margin bounding box for human crops to be properly detected
            MARGIN = 10

            num_people = 0  # 重置人数为0

            for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
                with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                    # Media pose prediction
                    cropped_image = image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN]
                    cropped_depth_image = depth_image[int(ymin) + MARGIN:int(ymax) + MARGIN,
                                          int(xmin) + MARGIN:int(xmax) + MARGIN]
                    cropped_image = self.findPose(cropped_image)
                    person_points, person_depth, sucess = self.findPosition(cropped_image, cropped_depth_image)
                    if sucess == 0: continue
                    show_points.append(person_points)
                    show_depth.append(person_depth)
                    img_list.append(cropped_image)
                    num_people += 1  # 每检测到一个人，人数加1

            pose_links = self.pose_link(num_people)

            if num_people > 0:
                self.show_points = torch.tensor(show_points, device="cuda:0", dtype=torch.float32)
                self.show_depth = torch.tensor(show_depth, device="cuda:0", dtype=torch.float32)

            t_show_depth = self.show_depth.clone()
            t_show_depth = t_show_depth.cuda(0)
            t_show_points = self.show_points.clone()
            t_show_points = t_show_points.cuda(0)
            #
            # print("depth", t_show_depth.size())
            # print("points", t_show_points.size())

            # Display the resulting image with person count
            cv2.putText(image, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Activity Recognition', image)
            # Write the frame to video file
            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if self.cfg['render_mode'] == 'centers':
                pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
                cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
            else:
                im, depth, sil = render(w2c, k, scene_data, scene_depth_data, self.cfg)
                if self.cfg['show_sil']:
                    im = (1 - sil).repeat(3, 1, 1)
                # if continue_flag == 1:
                #     pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
                print("shape",t_show_points.shape)
                pts, cols, pose_lineset, pose_colors, pose_points = ycy_rgbd2pcd(im, depth, w2c, k, self.cfg, t_show_points, t_show_depth, k_c2w, pose_links)

            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            pose_lines.lines = pose_lineset
            pose_lines.colors = pose_colors
            pose_lines.points = pose_points
            vis.update_geometry(pose_lines)

            if not vis.poll_events():
                break
            vis.update_renderer()

        # Cleanup
        pipeline.stop()
        out.release()
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

    detector = PoseDetector(scene_path, viz_cfg)

    detector.run()