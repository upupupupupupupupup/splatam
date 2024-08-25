#coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
#像素对齐了
pipeline = rs.pipeline()

#Create a config并配置要流​​式传输的管道
config = rs.config()
config.enable_device_from_file("/home/ycy/project/SplaTAM/data/dormitory/20221128_161053.bag")#这是打开相机API录制的视频
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# 保存路径
save_path = '/home/ycy/project/SplaTAM/data/dormitory/'

# 保存的图片和实时的图片界面
# cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
number = 0

file_handle1 = open('/home/ycy/project/SplaTAM/data/dormitory/rgb.txt', 'w')
file_handle2 = open('/home/ycy/project/SplaTAM/data/dormitory/depth.txt', 'w')

# 主循环
try:
    while True:
        #获得深度图的时间戳
        frames = pipeline.wait_for_frames()
        number = number + 1
        depth_timestamp = "%.6f" % (frames.timestamp / 1000)
        rgb_timestamp = "%.6f" % (frames.timestamp / 1000 + 0.000017)#对比了 提取图片.py 的时间戳，发现rgb比depth多0.000017

        aligned_frames = align.process(frames)
        #获取对齐后的深度图与彩色图
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
        
        #下面两行是opencv显示部分
        # cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))
        # key = cv2.waitKey(30)

        rgb_image_name = rgb_timestamp+ ".png"
        depth_image_name = depth_timestamp+ ".png"
        rgb_path = "rgb/" + rgb_image_name
        depth_path = "depth/" + depth_image_name
        # rgb图片路径及图片保存为png格式
        file_handle1.write(rgb_timestamp + " " + rgb_path + '\n')
        cv2.imwrite(save_path + rgb_path, color_image)
        # depth图片路径及图片保存为png格式
        file_handle2.write(depth_timestamp + " " + depth_path + '\n')
        cv2.imwrite(save_path + depth_path, depth_image)
        print(number, rgb_timestamp, depth_timestamp)
        #cv2.imshow("save", np.hstack((saved_color_image, saved_depth_mapped_image)))

        #查看话题包有多少帧图片，下面就改成多少
        if number == 2890:
            cv2.destroyAllWindows()
            break    
finally:
    pipeline.stop()

file_handle1.close()
file_handle2.close()

