# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http:# www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
from pyorbbecsdk import *
import cv2
import numpy as np
from orbbec_utils import frame_to_bgr_image
import sys
import argparse
import os
import matplotlib.pyplot as plt

ESC_KEY = 27

def save_depth_frame(frame: DepthFrame, img, index):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = "%.6f" %  frame.get_timestamp()
    scale = frame.get_depth_scale()
    #data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    #data = data.reshape((height, width))
    #data = data.astype(np.float32) * scale
    #data = data.astype(np.uint16)

    #data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #depth_image = data.astype(np.uint16)
    save_image_dir = os.path.join(os.getcwd(), "depth_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    raw_filename = save_image_dir + "/depth_{}x{}_{}_{}.png".format(width, height, index, timestamp)
    #data.tofile(raw_filename)
    cv2.imwrite(raw_filename, img)


def save_color_frame(frame: ColorFrame, img, index):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = frame.get_timestamp()
    save_image_dir = os.path.join(os.getcwd(), "color_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    filename = save_image_dir + "/color_{}x{}_{}_{}.png".format(width, height, index, timestamp)
    #image = frame_to_bgr_image(frame)
    if img is None:
        print("failed to convert frame to image")
        return
    cv2.imwrite(filename, img)


def main(argv):
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode",
                        help="align mode, HW=hardware mode,SW=software mode,NONE=disable align",
                        type=str, default='HW')
    parser.add_argument("-s", "--enable_sync", help="enable sync", type=bool, default=True)
    args = parser.parse_args()
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
        depth_profile = profile_list.get_video_stream_profile(640, 400, OBFormat.Y12, 30)
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

    saved_color_cnt: int = 0
    saved_depth_cnt: int = 0


    while True:
        try:
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
            #depth_data = depth_data.astype(np.float32) * scale
            #depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


            # 将深度图像转换为32位浮点数（CV_32F）类型
            cv_image_32f = depth_data.astype(np.float32) * scale

            # 将深度图像范围归一化到0-1之间
            cv_image_normalized = cv2.normalize(cv_image_32f, None, 0.0, 1.0, cv2.NORM_MINMAX)

            # 将深度图像扩展到0-65535范围，并转换为16位无符号整数（CV_16U）
            cv_image_mono16 = (cv_image_normalized * 65535).astype(np.uint16)

            # 获取深度图像尺寸
            # depth_height, depth_width = cv_image_mono16.shape

            if  saved_color_cnt < 5:
                save_color_frame(color_frame, color_image, saved_color_cnt)
                saved_color_cnt += 1
            if  saved_depth_cnt < 5:
                save_depth_frame(depth_frame, cv_image_mono16, saved_depth_cnt)
                saved_depth_cnt += 1

            #depth_image = cv2.applyColorMap(cv_image_mono16, cv2.COLORMAP_JET)
            # overlay color image on depth image
            #depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
            cv2.imshow("SyncAlignViewer ", cv_image_mono16)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
