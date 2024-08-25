from collections import defaultdict

import cv2
import numpy as np
import pyrealsense2 as rs

from ultralytics import YOLO

MIN_DEPTH = 200  # 20mm
MAX_DEPTH = 5000  # 10000mm

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# # Open the video file
# video_path = "path/to/video.mp4"
# cap = cv2.VideoCapture(video_path)

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

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while True:
    # # Read a frame from the video
    # success, frame = cap.read()

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

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(color_image, persist=True, classes=0, device=0)

    # Get the boxes and track IDs
    # boxes = results[0].boxes.xywh.cpu()
    # track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    # for box, track_id in zip(boxes, track_ids):
    #     x, y, w, h = box
    #     track = track_history[track_id]
    #     track.append((float(x), float(y)))  # x, y center point
    #     if len(track) > 30:  # retain 90 tracks for 90 frames
    #         track.pop(0)
    #
    #     # Draw the tracking lines
    #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
# cap.release()
cv2.destroyAllWindows()