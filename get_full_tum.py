#!/bin/python
import roslib
import rosbag
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Quaternion, Point
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError


ros_bag = 'image4.bag'  #bag包路径
save_path = '/home/agx/PycharmProjects/ycy_SplaTAM/'   #输出数据集的路径
rgb = save_path + 'rgb/'  #rgb path
depth = save_path + 'depth/'   #depth path

bridge = CvBridge()

file_handle1 = open(save_path + 'rgb.txt', 'w')
file_handle2 = open(save_path + 'depth.txt', 'w')
depth_num = 1
get_depth = False
#file_handle3 = open(save_path + 'groundtruth.txt', 'w')

with rosbag.Bag(ros_bag, 'r') as bag:
    for topic,msg,t in bag.read_messages():

        if topic == "/camera/depth/image_raw":  #depth topic
            cv_image = bridge.imgmsg_to_cv2(msg)
            #cv_image = bridge.imgmsg_to_cv2(msg, '32FC1')
            #cv_image = cv_image * 255
            #timestr = "%.6f" %  msg.header.stamp.to_sec()   #depth time stamp
            if depth_num % 5 == 0:
                get_depth = True
                timestr = "%.6f" % msg.header.stamp.to_sec()
                image_name = timestr+ ".png"
                path = "depth/" + image_name
                file_handle2.write(timestr + " " + path + '\n')
                cv2.imwrite(depth + image_name, cv_image)
            depth_num += 1
        if topic == "/camera/color/image_raw":   #rgb topic
            cv_image = bridge.imgmsg_to_cv2(msg,"bgr8")
            #timestr = "%.6f" %  msg.header.stamp.to_sec()   #rgb time stamp
            if get_depth == True:
                get_depth = False
                image_name = timestr+ ".png"
                path = "rgb/" + image_name
                file_handle1.write(timestr + " " + path + '\n')
                cv2.imwrite(rgb + image_name, cv_image)

        '''
        if topic == '/vrpn_client_node/RigidBody_ZLT_UAV/pose': #groundtruth topic
            p = msg.pose.position
            q = msg.pose.orientation
            timestr = "%.6f" %  msg.header.stamp.to_sec()
            file_handle3.write(timestr + " " + str(round(p.x, 4)) + " " + str(round(p.y, 4)) + " " + str(round(p.z, 4)) + " ")
            file_handle3.write(str(round(q.x, 4)) + " " + str(round(q.y, 4)) + " " + str(round(q.z, 4)) + " " + str(round(q.w, 4)) + '\n')
        '''
file_handle1.close()
file_handle2.close()
#file_handle3.close()


