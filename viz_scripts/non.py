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

import time
import onnxruntime as ort


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

    def findPosition(self, img, draw=True):
        # print(results.pose_landmarks)
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(
                    self.results.pose_landmarks.landmark):  # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                h, w, c = img.shape  # 返回图片的(高,宽,位深)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)  # lm.x  lm.y是比例  乘上总长度就是像素点位置
                lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # 画蓝色圆圈
        return lmList


detector = poseDetector()
lmLists = []
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

# cap = cv2.VideoCapture(1)  # 使用默认摄像头设备
cap = cv2.VideoCapture('q3.mp4')  # 视频

# 获取摄像头的尺寸
ret, frame = cap.read()
h, w, _ = frame.shape
size = (w, h)

# 创建用于保存视频的 VideoWriter 对象
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

num_people = 0  # 用于记录检测到的人数

# 构建两个实例，分别用于连接不同的监听端口
client1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client1.connect(('127.0.0.1', 9999))
client2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client2.connect(('127.0.0.1', 8888))

clients = {}  # 使用字典存储多个客户端套接字对象
clients[0] = client1
clients[1] = client2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Recolor Feed from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # making image writeable to False improves prediction
    image.flags.writeable = False

    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # This array will contain crops of images in case we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    num_people = 0  # 重置人数为0

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            # Media pose prediction
            cropped_image = image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN]
            cropped_image = detector.findPose(cropped_image)
            lmLists.append(detector.findPosition(cropped_image))
            img_list.append(cropped_image)

            num_people += 1  # 每检测到一个人，人数加1

            # 将关键点坐标映射回原始帧
            if lmLists[num_people - 1] is not None:
                for az, lm in enumerate(lmLists[num_people - 1]):
                    id, cx, cy, cz = lm  # 解包关键点信息
                    if cy < y0min:
                        y0min = cy

            if lmLists[num_people - 1] is not None:
                for i, lm in enumerate(lmLists[num_people - 1]):
                    id, cx, cy, cz = lm  # 解包关键点信息
                    cx = cx
                    cy = cy
                    cz = cz
                    # 更新lmLists中的关键点坐标
                    lmLists[num_people - 1][i] = [id, cx, cy, cz]

    # Display the resulting image with person count
    cv2.putText(image, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Activity Recognition', image)
    # Write the frame to video file
    out.write(image)

    if num_people == 2:
        # 执行你的操作，例如打印一条消息
        print("There are 2 people in the video.")
        i = 0

        # 将每个人的坐标点信息发送到不同的端口上
        for i in range(num_people):
            if len(lmLists[i]) != 0:
                for data in lmLists[i]:
                    print(data)  # print(lmList[n]) 可以打印第n个
                    for a in range(1, 4):
                        if a == 2:
                            strdata = strdata + str(frame.shape[0] - data[a]) + ','
                        else:
                            strdata = strdata + str(data[a]) + ','
                print(str(clients[i]) + ':' + strdata)
                clients[i].send(strdata.encode('utf-8'))
                strdata = ""

    lmLists = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()