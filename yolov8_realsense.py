import time
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from ultralytics import YOLO  # 将YOLOv8导入到该py文件中

''' 深度相机 '''
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流

pipe_profile = pipeline.start(config)  # streaming流开始
align = rs.align(rs.stream.color)


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    depth_colormap = cv2.applyColorMap \
        (cv2.convertScaleAbs(img_depth, alpha=0.008)
         , cv2.COLORMAP_JET)

    return depth_intrin, img_color, aligned_depth_frame


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可

    # 设置计时器
    start_time = time.time()
    interval = 6  # 间隔时间（秒）
    try:
        while True:
            depth_intrin, img_color, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
            # 检查是否达到间隔时间
            if time.time() - start_time >= interval:
                start_time = time.time()  # 重置计时器
                source = [img_color]

                # 调用YOLOv8中的推理，还是相当于把d435i中某一帧的图片进行detect推理
                results = model.predict(source, save=False, show_conf=False)

                for result in results:  # 相当于都预测完了才进行的打印目标框，这样就慢了
                    boxes = result.boxes.xywh.tolist()
                    im_array = result.plot()  # plot a BGR numpy array of predictions

                    for i in range(len(boxes)):
                        ux, uy = int(boxes[i][0]), int(boxes[i][1])  # 计算像素坐标系的x
                        dis = aligned_depth_frame.get_distance(ux, uy)
                        camera_xyz = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                        camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                        camera_xyz = np.array(list(camera_xyz)) * 1000
                        camera_xyz = list(camera_xyz)

                        cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                        cv2.putText(im_array, str(camera_xyz), (ux + 20, uy + 10), 0, 0.5,
                                    [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

                cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                                   cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow('detection', 640, 480)
                cv2.imshow('detection', im_array)
                cv2.waitKey(2000)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                break
    finally:
        # Stop streaming
        pipeline.stop()