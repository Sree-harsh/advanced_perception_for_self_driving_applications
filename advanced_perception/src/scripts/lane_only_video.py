#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from lanedetect_utils import initialize_lane_detection, preprocess_image, postprocess_image
import torch
import torchvision.transforms as transforms

class LaneDetector:

    def __init__(self):

        cap = cv2.VideoCapture('highway_nosound.mp4')
        self.img_w = 0
        self.img_h = 0
        self.row_anchor = []
        self.cfg = None
        self.cls_num_per_lane = 56

        self.initialize_detection()


        if not cap.isOpened():
            print("Error loading video")

        while cap.isOpened():
 
            ret, self.frame = cap.read()
        
            # If an image frame has been grabbed, display it
            if ret:
                # cv2.imshow("frame" , self.frame)
                self.process_image(self.frame)
                # print(self.frame.shape)

            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

        pass
    
    def initialize_detection(self):

        # getting the required variables value
        self.net, self.img_w, self.img_h, self.row_anchor , self.cfg = initialize_lane_detection()

        # Define image transformations for preprocessing
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def process_image(self, imgs):

        # Create a copy of the input image
        imgs = cv2.resize(imgs, (1280,720),interpolation=cv2.INTER_AREA)
        imgs_copy = imgs.copy()

        # Preprocess the image
        x = preprocess_image(imgs, self.img_transforms)

        # Perform inference on the input image
        with torch.no_grad():
            out = self.net(x)

        print(self.img_w)

        # Postprocess the image and obtain lane markers
        lane_markers, lane_markers_rgb = postprocess_image(out, imgs_copy, self.cfg, self.img_w, self.img_h, self.cls_num_per_lane, self.row_anchor)

        # Resize the lane markers image (if needed)
        lane_markers_rgb_resized = cv2.resize(lane_markers_rgb, (640, 480), interpolation=cv2.INTER_AREA)

        cv2.imshow("Frame2" , lane_markers_rgb_resized)        

if __name__ == "__main__":

    LaneDetector()
    


 

 

 
