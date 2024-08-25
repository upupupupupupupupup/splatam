import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import ast
import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import imageio
import numpy as np
import torch
import torch.optim as optim
from arguments import ModelParams, PipelineParams, get_combined_args, iComMaParams
from gaussian_renderer import GaussianModel, ycy_render
from scene import Scene
from scene.cameras import Camera_Pose
from ycy_utils.calculate_error_utils import cal_campose_error
from ycy_utils.common_utils import seed_everything
from ycy_utils.general_utils import safe_state
from ycy_utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from ycy_utils.icomma_helper import (
    get_pose_estimation_input,
    load_LoFTR,
    ycy_get_pose_estimation_input,
)
from ycy_utils.image_utils import to8b
from ycy_utils.loss_utils import loss_loftr, loss_mse


def load_camera(cfg, scene_path):
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


def camera_pose_estimation(
    params,
    background: torch.tensor,
    pipeline: PipelineParams,
    icommaparams: iComMaParams,
    icomma_info,
    output_path,
    scene_path,
    cfg,
):
    # start pose & gt pose
    # gt_pose_c2w=icomma_info.gt_pose_c2w

    w2c, intr = load_camera(cfg, scene_path)

    start_pose_w2c = torch.from_numpy(w2c).cuda()

    FoVx = focal2fov(intr[0, 0], icomma_info.image_width)
    FoVy = focal2fov(intr[1, 1], icomma_info.image_height)

    # query_image for comparing
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(
        start_pose_w2c,
        FoVx=FoVx,
        FoVy=FoVy,
        image_width=icomma_info.image_width,
        image_height=icomma_info.image_height,
    )
    camera_pose.cuda()

    # store gif elements
    imgs = []

    matching_flag = not icommaparams.deprecate_matching

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(), lr=icommaparams.camera_pose_lr)
    iter = icommaparams.pose_estimation_iter
    num_iter_matching = 0
    for k in range(iter):

        rendering = ycy_render(
            camera_pose,
            params,
            pipeline,
            background,
            compute_grad_cov2d=icommaparams.compute_grad_cov2d,
        )["render"]

        if matching_flag:
            loss_matching = loss_loftr(
                query_image,
                rendering,
                LoFTR_model,
                icommaparams.confidence_threshold_LoFTR,
                icommaparams.min_matching_points,
            )
            loss_comparing = loss_mse(rendering, query_image)

            if loss_matching is None:
                loss = loss_comparing
            else:
                loss = (
                    icommaparams.lambda_LoFTR * loss_matching
                    + (1 - icommaparams.lambda_LoFTR) * loss_comparing
                )
                if loss_matching < 0.001:
                    matching_flag = False

            num_iter_matching += 1
        else:
            loss_comparing = loss_mse(rendering, query_image)
            loss = loss_comparing

            new_lrate = icommaparams.camera_pose_lr * (
                0.6 ** ((k - num_iter_matching + 1) / 50)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate

        # output intermediate results
        if (k + 1) % 20 == 0 or k == 0:
            print("Step: ", k)
            if matching_flag and loss_matching is not None:
                print("Matching Loss: ", loss_matching.item())
            print("Comparing Loss: ", loss_comparing.item())
            print("Loss: ", loss.item())

            # record error

            with torch.no_grad():
                cur_pose_c2w = camera_pose.current_campose_c2w()
                # rot_error,translation_error=cal_campose_error(cur_pose_c2w,gt_pose_c2w)
                # print('Rotation error: ', rot_error)
                # print('Translation error: ', translation_error)
                print("current_c2w:\n", np.round(cur_pose_c2w, decimals=5))
                print("-----------------------------------")

            # output images
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k) + ".png")
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, "video.gif"), imgs, fps=2)  # fps=4


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    parser.add_argument("experiment", type=str, help="Path to experiment file")  # need
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--query_path", default="", type=str, help="query path")  # need
    parser.add_argument("--output_path", default="output", type=str, help="output path")
    parser.add_argument("--obs_img_index", default=0, type=int)
    parser.add_argument("--delta", default="[1,1,1,1,1,1]", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    # args = get_combined_args(parser)

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

    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {
        k: torch.tensor(all_params[k]).cuda().float().requires_grad_(True)
        for k in all_params.keys()
    }
    # intrinsics = torch.tensor(intrinsics).cuda().float()
    # first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

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

    # Initialize system state (RNG)
    safe_state(args.quiet)

    makedirs(args.output_path, exist_ok=True)

    # load LoFTR_model
    LoFTR_model = load_LoFTR(
        icommaparams.LoFTR_ckpt_path, icommaparams.LoFTR_temp_bug_fix
    )

    # load gaussians
    # dataset = model.extract(args)
    # gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information.
    # You can customize the iComMa_input_info in practical applications.
    # scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=True)
    print("index", args.obs_img_index)
    # obs_view=scene.getTestCameras()[args.obs_img_index]
    # obs_view = scene.getTrainCameras()[args.obs_img_index]

    # icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    icomma_info = ycy_get_pose_estimation_input(args)

    # pose estimation
    camera_pose_estimation(
        params,
        background,
        pipeline,
        icommaparams,
        icomma_info,
        args.output_path,
        scene_path,
        viz_cfg,
    )
