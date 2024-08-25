import multiprocessing
import time

import os
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import socket
# PyTorch Hub
import torch
import cv2
import mediapipe as mp
import torch.nn as nn

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering

MIN_DEPTH = 200  # 20mm
MAX_DEPTH = 5000  # 10000mm

n_l = 2
num = (2*n_l+1)**2

pose_keypoints_id = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

#pose_link = np.array([[0, 1], [1, 7], [7, 6], [6, 0], [1, 3], [3, 5], [0, 2], [2, 4], [6, 8], [8, 10], [7, 9], [9, 11]])


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
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
        if self.results.pose_landmarks:
            l_hip_x = int(w * self.results.pose_landmarks.landmark[23].x)
            l_hip_y = int(h * self.results.pose_landmarks.landmark[23].y)
            r_hip_x = int(w * self.results.pose_landmarks.landmark[24].x)
            r_hip_y = int(h * self.results.pose_landmarks.landmark[24].y)
            if l_hip_x > 0 and l_hip_y > 0 and r_hip_x > 0 and r_hip_y > 0 and l_hip_x < w and l_hip_y < h and r_hip_x < w and r_hip_y < h:
                if depth[l_hip_y, l_hip_x] > 1e-5 and depth[r_hip_y, r_hip_x] > 1e-5 :
                    center_depth = (depth[int(l_hip_y), int(l_hip_x)] + depth[int(r_hip_y), int(r_hip_x)]) / 2000.0
                    for id, lm in enumerate(self.results.pose_landmarks.landmark):  # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z + center_depth)  # lm.x  lm.y是比例  乘上总长度就是像素点位置
                        cx = max(min(cx,w-n_l-1),n_l)
                        cy = max(min(cy,h-n_l-1),n_l)
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # 画蓝色圆圈
                        if id in pose_keypoints_id:  # only save keypoints that are indicated in pose_keypoints
                            for m in range(-n_l, n_l + 1):
                                for n in range(-n_l, n_l + 1):
                                    show_points.append([cx + m, cy + n])
                                    show_depth.append(cz)
        return show_points, show_depth

def do_socket(conn, addr, ):
    try:
        detector = poseDetector()
        # lmLists = []
        strdata = ""
        y0min = 100000

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

        mp_drawing = mp.solutions.drawing_utils
        # mp_pose = mp.solutions.pose     detector.findPose

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
        while True:
            if conn.poll(1) == False:
                time.sleep(0.5)
                continue
            data = conn.recv()  # 等待接受数据
            print(data)
            conn.send('sucess')
            # ***********************
            # 要执行的程序写在这里
            # ***********************
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
                    cropped_image = detector.findPose(cropped_image)
                    # lmLists.append(detector.findPosition(cropped_image, depth_image))
                    person_points, person_depth = detector.findPosition(cropped_image, cropped_depth_image)
                    show_points.append(person_points)
                    show_depth.append(person_depth)
                    img_list.append(cropped_image)
                    n_p = 12 * num * num_people
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

                    num_people += 1  # 每检测到一个人，人数加1

            # Display the resulting image with person count
            cv2.putText(image, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Activity Recognition', image)
            # Write the frame to video file
            out.write(image)
            conn.send(show_points,show_depth,use_lines)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print('Socket Error', e)

    finally:
        try:
            conn.close()
            print('Connection close.', addr)
        except:
            print('close except')


def run_server(host, port):
    from multiprocessing.connection import Listener
    server_sock = Listener((host, port))

    print("Sever running...", host, port)

    pool = multiprocessing.Pool(10)
    while True:
        # 接受一个新连接:

        conn = server_sock.accept()
        addr = server_sock.last_accepted
        print('Accept new connection', addr)

        # 创建进程来处理TCP连接:
        pool.apply_async(func=do_socket, args=(conn, addr,))


if __name__ == '__main__':
    server_host = '127.0.0.1'
    server_port = 8000
    run_server(server_host, server_port)
