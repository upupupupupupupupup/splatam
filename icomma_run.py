import os
import sys
from importlib.machinery import SourceFileLoader
#error please use new_run.py
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
from scene import Scene
import torch.optim as optim
from os import makedirs
from ycy_utils.general_utils import safe_state
from ycy_utils.calculate_error_utils import cal_campose_error
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from ycy_utils.icomma_helper import load_LoFTR, get_pose_estimation_input, ycy_get_pose_estimation_input
from ycy_utils.image_utils import to8b
from ycy_utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

import cv2
import imageio
import numpy as np
import math
import ast
from scene.cameras import Camera_Pose
from ycy_utils.loss_utils import loss_loftr,loss_mse

# from diff_gaussian_rasterization import GaussianRasterizer
# from diff_gaussian_rasterization import GaussianRasterizationSettings

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from ycy_utils.common_utils import seed_everything
from ycy_utils.recon_helpers import setup_camera
from ycy_utils.slam_helpers import get_depth_and_silhouette
from ycy_utils.slam_external import build_rotation

from gaussian_renderer import ycy_render

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


def load_scene_data(viewpoint_camera, scene_path, first_frame_w2c, intrinsics):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float().requires_grad_(True) for k in all_params.keys()}
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

    # all_w2cs = []
    # num_t = params['cam_unnorm_rots'].shape[-1]
    # for t_i in range(num_t):
    #     cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
    #     cam_tran = params['cam_trans'][..., t_i]
    #     rel_w2c = torch.eye(4).cuda().float()
    #     rel_w2c[:3, :3] = build_rotation(cam_rot)
    #     rel_w2c[:3, 3] = cam_tran
    #     all_w2cs.append(rel_w2c.cpu().numpy())

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
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
        'camera_center': viewpoint_camera.camera_center,
        'camera_pose': viewpoint_camera.world_view_transform
    }
    # depth_rendervar = {
    #     'means3D': params['means3D'],
    #     'colors_precomp': get_depth_and_silhouette(params['means3D'], first_frame_w2c),
    #     'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
    #     'opacities': torch.sigmoid(params['logit_opacities']),
    #     'scales': torch.exp(log_scales),
    #     'means2D': torch.zeros_like(params['means3D'], device="cuda")
    # }
    return rendervar, params

def render(viewpoint_camera, w2c, k, timestep_data, cfg):
    # cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    white_bg_cam = Camera(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        compute_grad_cov2d=True,
        proj_k=viewpoint_camera.projection_matrix
    )
    timestep_data['means2D'].retain_grad()
    im, _, depth, _, _, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
    #depth_sil, _, _, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
    #differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
    #sil = depth_sil[1, :, :].unsqueeze(0)
    return im, depth

# def ycy_render(params, viewpoint_camera, bg_color: torch.Tensor, scaling_modifier=1.0, compute_grad_cov2d=True):
#     """
#     Render the scene.
#
#     Background tensor (bg_color) must be on GPU!
#     """
#
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     screenspace_points = torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass
#
#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=0,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         compute_grad_cov2d=compute_grad_cov2d,
#         proj_k=viewpoint_camera.projection_matrix
#     )
#
#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#
#     # means3D = pc.get_xyz
#     # means2D = screenspace_points
#     # opacity = pc.get_opacity
#
#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     # if pipe.compute_cov3D_python:
#     #     cov3D_precomp = pc.get_covariance(scaling_modifier)
#     # else:
#     #     scales = pc.get_scaling
#     #     rotations = pc.get_rotation
#
#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#
#     # shs = None
#     # colors_precomp = None
#     # if override_color is None:
#     #     if pipe.convert_SHs_python:
#     #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
#     #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#     #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
#     #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#     #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#     #     else:
#     #         shs = pc.get_features
#     # else:
#     #     colors_precomp = override_color
#
#     # Check if Gaussians are Isotropic
#     if params['log_scales'].shape[1] == 1:
#         log_scales = torch.tile(params['log_scales'], (1, 3))
#     else:
#         log_scales = params['log_scales']
#
#     # Rasterize visible Gaussians to image, obtain their radii (on screen).
#     rendered_image, radii, _, _, _, = rasterizer(
#         means3D=params['means3D'],
#         means2D=screenspace_points,
#         colors_precomp=params['rgb_colors'],
#         opacities=torch.sigmoid(params['logit_opacities']),
#         scales=torch.exp(log_scales),
#         rotations=torch.nn.functional.normalize(params['unnorm_rotations']),
#         camera_center=viewpoint_camera.camera_center,
#         camera_pose=viewpoint_camera.world_view_transform)
#
#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter": radii > 0,
#             "radii": radii}

def camera_pose_estimation(background:torch.tensor, icommaparams:iComMaParams, icomma_info, output_path, scene_path, cfg):
    # start pose & gt pose
    #gt_pose_c2w=icomma_info.gt_pose_c2w

    w2c, intr = load_camera(cfg, scene_path)

    start_pose_w2c = torch.from_numpy(w2c).cuda()

    FoVx = focal2fov(intr[0, 0], icomma_info.image_width)
    FoVy = focal2fov(intr[1, 1], icomma_info.image_height)

    # query_image for comparing
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c, FoVx=FoVx, FoVy=FoVy,
                              image_width=icomma_info.image_width, image_height=icomma_info.image_height)
    camera_pose.cuda()

    scene_data, params= load_scene_data(camera_pose, scene_path, w2c, intr)

    # params = dict(np.load(scene_path, allow_pickle=True))
    # params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}

    #start_pose_w2c=icomma_info.start_pose_w2c.cuda()

    # store gif elements
    imgs=[]
    
    matching_flag= not icommaparams.deprecate_matching

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    iter = icommaparams.pose_estimation_iter
    num_iter_matching = 0
    for k in range(iter):
        im = ycy_render(params,camera_pose,background,compute_grad_cov2d = icommaparams.compute_grad_cov2d)["render"]
        #im, depth = render(camera_pose, start_pose_w2c, intr, scene_data, cfg)
        # transformed_pts = transform_to_frame(params, iter_time_idx,
        #                                      gaussians_grad=False,
        #                                      camera_grad=True)
        # rendervar = transformed_params2rendervar(params, transformed_pts)
        # im, radius, _, _, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)

        if matching_flag:
            loss_matching = loss_loftr(query_image,im,LoFTR_model,icommaparams.confidence_threshold_LoFTR,icommaparams.min_matching_points)
            loss_comparing = loss_mse(im,query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                loss = icommaparams.lambda_LoFTR *loss_matching + (1-icommaparams.lambda_LoFTR)*loss_comparing
                if loss_matching<0.001:
                    matching_flag=False
                    
            num_iter_matching += 1
        else:
            loss_comparing = loss_mse(im,query_image)
            loss = loss_comparing
            
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

        #loss.requires_grad_(True)
        
        # output intermediate results
        if (k + 1) % 20 == 0 or k == 0:
            print('Step: ', k)
            if matching_flag and loss_matching is not None:
                print('Matching Loss: ', loss_matching.item())
            print('Comparing Loss: ', loss_comparing.item())
            print('Loss: ', loss.item())

            # record error

            with torch.no_grad():
                cur_pose_c2w = camera_pose.current_campose_c2w()
                print("current_c2w:\n", cur_pose_c2w)
                print('-----------------------------------')

            # output images
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = im.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=4) #fps=4
  
if __name__ == "__main__":
    # Set up command line argument parser

    parser = ArgumentParser(description="Camera pose estimation parameters")
    #model = ModelParams(parser, sentinel=True)
    #pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--query_path", default='', type=str,help="query path")
    parser.add_argument("--output_path", default='output', type=str,help="output path")
    parser.add_argument("--obs_img_index", default=0, type=int)
    parser.add_argument("--delta", default="[30,10,10,0.1,0.1,0.1]", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    #args = get_combined_args(parser)
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
   
    # Initialize system state (RNG)
    safe_state(args.quiet)

    makedirs(args.output_path, exist_ok=True)
    
    # load LoFTR_model
    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path,icommaparams.LoFTR_temp_bug_fix)
    
    # load gaussians
    #dataset = model.extract(args)
    #gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    #scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=True)
    print("index",args.obs_img_index)
    #obs_view=scene.getTestCameras()[args.obs_img_index]
    #obs_view=scene.getTrainCameras()[args.obs_img_index]

    #icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    icomma_info=ycy_get_pose_estimation_input(args)
    
    # pose estimation
    camera_pose_estimation(background, icommaparams,icomma_info,args.output_path,scene_path,viz_cfg)