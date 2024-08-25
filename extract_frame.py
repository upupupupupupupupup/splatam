import pyrealsense2 as rs
from os import makedirs
import numpy as np
import cv2
from os.path import join
import json
from utils import *

FILTERING = True
if FILTERING:
    SPATIAL_FILTER = True
    TEMPOTAL_FILTER = True

DECIMATION = 2 # 0,1,2,4,8
HOLE_FILLING = False

SAVE_IMAGES = True
DATA_PATH="../data"

def save_intrinsic_as_json(filename, frame, depth_scale, angle, max_d):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, intrinsics.ppx,
                    0, intrinsics.fy, intrinsics.ppy,
                    0, 0, 1
                ],
                'depth_scale':depth_scale,
                'angle':angle,
                'max_d':max_d
            },
            outfile,
            indent=4)

def save_selected_imgs(path, colors, depths, plants, rev):
    c_path = path +'color/'
    d_path = path+'depth/'
    # shutil.rmtree(c_path)
    makedirs(c_path)
    # shutil.rmtree(d_path)
    makedirs(d_path)

    for i, (color, depth) in enumerate(zip(colors, depths)):
        if rev:
            fc2 = plants[0] - i
        else:
            fc2 = plants[0] + i
        cv2.imwrite("%s/%06d.png" % \
                    (d_path, fc2), depth)
        cv2.imwrite("%s/%06d.jpg" % \
                    (c_path, fc2), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

def load_rosbag(bag_file):
    # =========== initialise pipeline ===============
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(f"{DATA_PATH}/bag_files/{bag_file}.bag", repeat_playback=False)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    if DECIMATION:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, DECIMATION)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    # ============= load jsondata ==============
    info = json.load(open(f"{DATA_PATH}/bag_files/{bag_file}.json"))["rgbd"]

    file = info["file"]
    angle = info["angle"]
    max_d = info["max_d"]
    frame_range = info["range"]
    n_frames = info["n_frames"]
    reverse = info["direction"]

    # if plants[0] > plants [1]:
    #     reverse_order = True
    #     #plants = [30, 12]
    #     EOF = plants[0] - plants[1]
    # else:
    #     reverse_order = False
    #     #plants = [12, 30]
    #     EOF = plants[1] - plants[0]
    EOF = n_frames
    offset = np.floor((frame_range[1]-frame_range[0]) / n_frames).astype(np.int64)


    #============= create folder structure =================
    path_output = f"{DATA_PATH}/image_files/{bag_file}"
    path_depth = join(path_output, "depth")
    path_color = join(path_output, "color")

    if SAVE_IMAGES:
        make_clean_folder(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)

    frame_count = 0
    frame_id = 0

    for i in range(frame_range[0]):
        frames = pipeline.wait_for_frames()
    while True:
        try:
            frames = pipeline.wait_for_frames()

            if frame_count%offset == 0:
                #frame_id +=1
                print("------ frame count: ", frame_count, "   frame Id: ",frame_id,"  --------------")
                #-------------- filtering ----------------

                if DECIMATION:
                    frames = decimation.process(frames).as_frameset()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame().as_video_frame()
                color_frame = aligned_frames.get_color_frame().as_video_frame()

                # Post processing filters by realsense
                if FILTERING:
                    aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
                    if SPATIAL_FILTER:
                        aligned_depth_frame = spatial.process(aligned_depth_frame)
                    if TEMPOTAL_FILTER:
                        aligned_depth_frame = temporal.process(aligned_depth_frame)
                    aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
                if HOLE_FILLING:
                    aligned_depth_frame = hole_filling.process(aligned_depth_frame)

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    print("Frame not found")

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # imshow([color_image1,depth_image1, color_image, depth_image], ['','','',''], False)
                #intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

                print(f"----frame {frame_count} loaded as {frame_id}----")

                if SAVE_IMAGES:
                    image_path = f"{DATA_PATH}/image_files/{bag_file}"
                    if frame_id == 1:
                        save_intrinsic_as_json(f"{image_path}/intrinsics.json" ,frames, depth_scale, angle, max_d)
                    if reverse:
                        fc2 = n_frames - frame_id
                    else:
                        fc2 = frame_id
                    cv2.imwrite("%s/%06d.png" % \
                                (f"{image_path}/depth", fc2), depth_image)
                    cv2.imwrite("%s/%06d.jpg" % \
                                (f"{image_path}/color", fc2), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                    print(f"----frame {fc2} saved ----")
                frame_id+=1
            frame_count+=1

            if EOF!=-1 and frame_id > EOF:
                break
        except(Exception):
            print("------- Error Occured ---------")
            break
    return True

if __name__ == '__main__':
    load_rosbag("R2_G3_R_10_12")
    # load_rosbag("test")