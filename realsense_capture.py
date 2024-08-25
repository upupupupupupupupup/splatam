## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import os

# Import OpenCV for easy image rendering
import cv2
import mediapipe as mp

# Import Numpy for easy array manipulation
import numpy as np

# First import the library
import pyrealsense2 as rs

MIN_DEPTH = 300  # 20mm
MAX_DEPTH = 10000  # 10000mm
Capture_Num = 150

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def save_depth_frame(img, timestamp):
    save_image_dir = os.path.join(os.getcwd(), "depth")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    raw_filename = save_image_dir + "/{}.png".format(timestamp)
    # data.tofile(raw_filename)
    cv2.imwrite(raw_filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_color_frame(img, timestamp):
    save_image_dir = os.path.join(os.getcwd(), "rgb")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    filename = save_image_dir + "/{}.png".format(timestamp)
    if img is None:
        print("failed to convert frame to image")
        return
    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def main():
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
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_device("234222304656")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    saved_cnt: int = 0
    num: int = 0

    temporal = rs.temporal_filter()

    # pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Streaming loop
    try:
        while saved_cnt < Capture_Num:

            num += 1

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

            aligned_depth_frame = temporal.process(aligned_depth_frame)

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_image = np.where(
                (depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH), depth_image, 0
            )

            # timestamp = "%.3f" % (aligned_depth_frame.get_timestamp() / 1000)

            if num % 10 == 0:
                save_depth_frame(depth_image, saved_cnt)
                save_color_frame(color_image, saved_cnt)
                saved_cnt += 1

            # # Remove background - Set pixels further than clipping_distance to grey
            # grey_color = 153
            # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            #
            # # Render images:
            # #   depth align to color on left
            # #   depth on right
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = np.hstack((bg_removed, depth_colormap))
            #
            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example', images)

            # Draw the pose annotation on the image.

            # results = pose.process(color_image)
            # mp_drawing.draw_landmarks(
            #     color_image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            # )
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Pose", cv2.flip(color_image, 1))

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        print("\nget_all_image")
        pipeline.stop()


if __name__ == "__main__":
    main()
