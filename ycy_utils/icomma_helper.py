import os
import sys
from PIL import Image
import torch
from LoFTR.src.loftr import LoFTR, default_cfg
from copy import deepcopy
import numpy as np
from typing import NamedTuple
from ycy_utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from ycy_utils.general_utils import PILtoTorch

# Load the pre-trained LoFTR model. For more details, please refer to https://github.com/zju3dv/LoFTR.
def load_LoFTR(ckpt_path:str,temp_bug_fix:bool):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = temp_bug_fix  # set to False when using the old ckpt
   
    LoFTR_model = LoFTR(config=_default_cfg)
    LoFTR_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    LoFTR_model= LoFTR_model.eval().cuda()
    
    return LoFTR_model

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_c2w.T
    RT_w2c[:3, 3] = T_w2c
    RT_c2w=np.linalg.inv(RT_w2c)
    return RT_c2w

def get_pose_estimation_input(obs_view,delta):
    gt_pose_c2w=combine_3dgs_rotation_translation(obs_view.R,obs_view.T)
    start_pose_c2w =  trans_t_xyz(delta[3],delta[4],delta[5]) @ rot_phi(delta[0]/180.*np.pi) @ rot_theta(delta[1]/180.*np.pi) @ rot_psi(delta[2]/180.*np.pi)  @ gt_pose_c2w
    icomma_info = iComMa_input_info(#gt_pose_c2w=gt_pose_c2w,
        start_pose_w2c=torch.from_numpy(np.linalg.inv(start_pose_c2w)).float(),
        query_image= obs_view.original_image[0:3, :, :],
        FoVx=obs_view.FoVx,
        FoVy=obs_view.FoVy,
        image_width=obs_view.image_width,
        image_height=obs_view.image_height)
    
    return icomma_info


def ycy_get_pose_estimation_input(args):
    # gt_pose_c2w=combine_3dgs_rotation_translation(obs_view.R,obs_view.T)
    data_device = torch.device("cuda")
    #image_path = os.path.join(images_folder, os.path.basename(extr.name))
    #image_name = os.path.basename(image_path).split(".")[0]
    image = Image.open(args.query_path)
    orig_w, orig_h = image.size
    # resolution_scale = 1.0
    # if args.resolution in [1, 2, 4, 8]:
    #     resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    # else:  # should be a type that converts to float
    #     if args.resolution == -1:
    #         if orig_w > 1600:
    #             global WARNED
    #             if not WARNED:
    #                 print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
    #                     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
    #                 WARNED = True
    #             global_down = orig_w / 1600
    #         else:
    #             global_down = 1
    #     else:
    #         global_down = orig_w / args.resolution
    #
    #     scale = float(global_down) * float(resolution_scale)
    resolution = (orig_w, orig_h)
    resized_image_rgb = PILtoTorch(image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    original_image = gt_image.clamp(0.0, 1.0).to(data_device)
    image_width = original_image.shape[2]
    image_height = original_image.shape[1]
    original_image *= torch.ones((1, image_height, image_width), device=data_device)
    '''
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    focal_length_x = cam_intrinsics.params[0]
    focal_length_y = cam_intrinsics.params[1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
    '''
    #start_pose_c2w = trans_t_xyz(delta[3], delta[4], delta[5]) @ rot_phi(delta[0] / 180. * np.pi) @ rot_theta(delta[1] / 180. * np.pi) @ rot_psi(delta[2] / 180. * np.pi)  # @ gt_pose_c2w

    icomma_info = iComMa_input_info(
        query_image=original_image[0:3, :, :],
        image_width=image_width,
        image_height=image_height)

    return icomma_info

class iComMa_input_info(NamedTuple):
    query_image:torch.tensor
    image_width:int
    image_height:int

    