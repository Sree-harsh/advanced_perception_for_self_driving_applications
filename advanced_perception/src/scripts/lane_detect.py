#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image as ros_img
from cv_bridge import CvBridge
import numpy as np
from lanedetect_utils import initialize_lane_detection, preprocess_image, postprocess_image
import torch
import torchvision.transforms as transforms


class LaneDetector:

    def __init__(self):

        # Initialize ROS node
        rospy.init_node('lane_detector', anonymous=True)

        # Subscribe to image topics
        rospy.Subscriber("/camera/depth_new/camera_img", ros_img, self.img_updater)
        rospy.Subscriber("/camera/depth_new/camera_depth", ros_img, self.img_updater_depth)


        # Create publishers for lane markers
        self.pub = rospy.Publisher('/camera/color/lane_markers', ros_img, queue_size=1)
        self.pub_rgb = rospy.Publisher('/camera/depth_new/lane_markers_rgb',ros_img, queue_size=1)

        # Use for Debugging the output of the model on live image
        # self.pub_rgb = rospy.Publisher('/camera/depth_new/lane_markers_rgb', ros_img, queue_size=1)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize image variables
        self.rgb_image = None
        self.time_stamp = None
        self.img_frame_id = "map_new"

        # Initialize image dimensions and lane detection parameters
        self.img_w = 0
        self.img_h = 0
        self.row_anchor = []
        self.cfg = None
        self.cls_num_per_lane = 56
        self.depth_image_h=0
        self.depth_image_w=0

        # Initialize lane detection
        self.initialize_detection()

        # Set the publishing rate
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.rgb_image is None:
                return
            # Process the image and publish lane markers
            self.process_image(self.rgb_image)
            rate.sleep()

    def initialize_detection(self):

        # getting the required variables value
        self.net, self.img_w, self.img_h, self.row_anchor , self.cfg = initialize_lane_detection()

        # Define image transformations for preprocessing
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def img_updater(self, data):

        # Callback function for updating the RGB image
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.time_stamp = data.header.stamp
        self.rgb_image = img

    def img_updater_depth(self, data):

        # Callback function for updating the RGB image
        img = self.bridge.imgmsg_to_cv2(data, "16UC1")
        # self.time_stamp = data.header.stamp
        # self.rgb_image = img
        # print(img.shape)
        self.depth_image_w = img.shape[0]
        self.depth_image_h = img.shape[1]

    def process_image(self, imgs):

        # Create a copy of the input image
        imgs = cv2.resize(imgs, (1280,720),interpolation=cv2.INTER_AREA)
        imgs_copy = imgs.copy()

        # Preprocess the image
        x = preprocess_image(imgs, self.img_transforms)

        # Perform inference on the input image
        with torch.no_grad():
            out = self.net(x)

        # Postprocess the image and obtain lane markers
        lane_markers, lane_markers_rgb = postprocess_image(out, imgs_copy, self.cfg, self.img_w, self.img_h, self.cls_num_per_lane, self.row_anchor)

        # Resize the lane markers image for publishing
        lane_markers_resized = cv2.resize(lane_markers, (self.depth_image_h, self.depth_image_w), interpolation=cv2.INTER_AREA)
        lane_markers_rgb_resized = cv2.resize(lane_markers_rgb, (self.depth_image_h, self.depth_image_w), interpolation=cv2.INTER_AREA)

        # Convert the lane markers image to a ROS image message
        imgs_ros = self.bridge.cv2_to_imgmsg(lane_markers_resized, encoding="bgr8")
        imgs_ros_rgb = self.bridge.cv2_to_imgmsg(lane_markers_rgb_resized, encoding="bgr8")

        # Set the timestamps for the ROS message
        imgs_ros.header.stamp = self.time_stamp
        imgs_ros_rgb.header.stamp = self.time_stamp

        # Set the frame ids for the ROS messages
        imgs_ros.header.frame_id  = self.img_frame_id
        imgs_ros_rgb.header.frame_id  = self.img_frame_id
        
        # Publish the lane marker images
        self.pub.publish(imgs_ros)
        self.pub_rgb.publish(imgs_ros_rgb)

if __name__ == '__main__':
    try:
        LaneDetector()

    except rospy.ROSInterruptException:
        pass
