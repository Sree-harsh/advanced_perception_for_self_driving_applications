#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.append("/home/sree/sree_ws/src/advanced_perception_for_self-driving_applications/advanced_perception/src/dl_model")
from model.model import parsingNet
from data.constant import tusimple_row_anchor

sys.path.append("/home/sree/sree_ws/src/advanced_perception_for_self-driving_applications/advanced_perception/src/dl_model/utils")
from common import merge_config
from dist_utils import dist_print

import scipy.special
import numpy as np
from PIL import Image
import cv2

def initialize_lane_detection():

    # Merge configuration options
    args, cfg = merge_config()
    dist_print('start testing...')

    # Check backbone configuration
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    # Set the number of classes per lane
    cls_num_per_lane = 56

    # Create the lane detection model
    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    # Load the model's weights
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']

    # Prepare the model's state dictionary
    compatible_state_dict = {}

    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    # Load the model's state dictionary
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # Set image dimensions and row anchor points
    img_w, img_h = 1280 , 720
    row_anchor = tusimple_row_anchor

    return net, img_w, img_h, row_anchor , cfg

def preprocess_image(img, img_transforms):

    # Convert the input image to a PIL image
    img = Image.fromarray(img)

    # Apply image transformations for preprocessing
    x = img_transforms(img)

    # Add a batch dimension and move the data to the GPU
    x = x.unsqueeze(0).cuda() + 1

    return x

def postprocess_image(out, imgs_copy, cfg, img_w, img_h, cls_num_per_lane, row_anchor):

    # Define column sample points for lane detection
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    # Process the model's output
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc

    # Initialize an empty black image for lane markers
    imgs_black = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # cv2.imshow("Frame inside",imgs_copy)
    # cv2.imshow("Frame outside",imgs_black)
    # cv2.waitKey(10)

    # Iterate over detected lane points
    for i in range(out_j.shape[1]):

        if np.sum(out_j[:, i] != 0) > 2:

            for k in range(out_j.shape[0]):

                if out_j[k, i] > 0:
                    
                    # Calculate lane marker position
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                            int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    
                    # Draw a circle at the detected lane marker position
                    cv2.circle(imgs_black, ppp, 5, (0, 255, 0), -1)
                    cv2.circle(imgs_copy,ppp,5,(0,255,0),-1)

    return imgs_black, imgs_copy


