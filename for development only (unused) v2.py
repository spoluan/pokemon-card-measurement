# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:12:52 2022

@author: Sevendi Eldrige Rifki Poluan
    # This file is used for development only
"""

import os
os.chdir("D:\\post\\pokemon-card-measurement")  
import cv2    
import numpy as np
import pandas as pd
import math
from FileUtils import FileUtils
from RotationUtils import RotationUtils
from ContourDetectionUtils import ContourDetectionUtils
from LineUtils import LineUtils
from CardRemovalUtils import CardRemovalUtils
from CardDrawUtils import CardDrawUtils

def draw(image=None, is_cnt=False, cnts=[], coordinates=[]):
    if is_cnt:
        whiteFrame = []
        if type(cnts) != dict:
            whiteFrame = 255 * np.ones(image.shape, np.uint8)
            try:
                
                whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), -1)
            except:
                whiteFrame = cv2.drawContours(whiteFrame, np.expand_dims(np.array(cnts[1], dtype=np.int32), axis=0), -1, (0, 0, 0), -1)
        else:
            whiteFrame = 255 * np.ones(image.shape, np.uint8)
            for key, val in cnts.items():
                whiteFrame = cv2.drawContours(whiteFrame, val, -1, (0, 0, 0), -1)
        return whiteFrame
    else:
        for coordinate in coordinates:
            xy = list(map(lambda x: int(x), coordinate)) 
            image = cv2.circle(image, xy, radius=8, color=(232, 54, 39), thickness=5)
        return image
     

fileUtils = FileUtils()  
rotationUtils = RotationUtils()
contourDetectionUtils = ContourDetectionUtils()
lineUtils = LineUtils()
cardRemovalUtils = CardRemovalUtils()
cardDrawUtils = CardDrawUtils() 

addr = './Datasets/data' 
addr_to_save = './outputs' 
img_path = 'normal_4.jpg'

image = cv2.imread(os.path.join(addr, img_path), 1)
rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

image, central = rotationUtils.image_auto_rotation(image)

cnts, top, bottom, right, left, \
    one_top, two_bottom, one_right, two_left, \
    top_line, bottom_line, right_line, left_line, central = lineUtils.get_outermost_line_to_measure(image, central, is_ouside=True, card_type=img_path)

# Testing
rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
image_with_point = draw(image=rgb, is_cnt=False, coordinates=[top_line, bottom_line, right_line, left_line, central])
cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_point)

image_1, removed_b_box = cardRemovalUtils.remove_card_center(image, central)
cv2.imwrite(os.path.join(addr_to_save, img_path), image_1)

seg_img, new_removed_b_box = cardRemovalUtils.remove_some_slices_from_the_edge(image_1, top_line, bottom_line, right_line, left_line, cut_control=20)
cv2.imwrite(os.path.join(addr_to_save, img_path), seg_img)

whiteFrameWithCountour, cnts, gap_removal = contourDetectionUtils.get_contours_inner(seg_img)
cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrameWithCountour) 

whiteFrames = cardRemovalUtils.frame_adjustment_to_the_original_size(image_1.shape, whiteFrameWithCountour, removed_b_box, new_removed_b_box, gap_removal)
cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrames) 

cnts, top, bottom, right, left, \
    one_top, two_bottom, one_right, two_left, \
    top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate, _ = lineUtils.get_outermost_line_to_measure(whiteFrames, central, is_binary=True, is_ouside=False, card_type=img_path)

# Testing
rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
image_with_point = draw(image=rgb, is_cnt=False, coordinates=[top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate, top_line, bottom_line, right_line, left_line, central])
cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_point) 

# whiteFrame = 255 * np.ones(image.shape, np.uint8)
# for x in sorted(one_right.keys()):
#     whiteFrame = draw(image=whiteFrame, cnts=np.expand_dims(np.array(one_right[x][1], dtype=np.int32), axis=1), is_cnt=True)
# cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrame) 
