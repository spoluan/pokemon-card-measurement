# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:23:53 2022

@author: Sevendi Eldrige Rifki Poluan
     # This file is used for development only
"""

import os
os.chdir("D:\\Card-contour-detection")  
import cv2    
import numpy as np
from FileUtils import FileUtils
from RotationUtils import RotationUtils
from ContourDetectionUtils import ContourDetectionUtils
from LineUtils import LineUtils
from CardRemovalUtils import CardRemovalUtils
from CardDrawUtils import CardDrawUtils

fileUtils = FileUtils()  
rotationUtils = RotationUtils() 
contourDetectionUtils = ContourDetectionUtils()
lineUtils = LineUtils()
cardRemovalUtils = CardRemovalUtils()
cardDrawUtils = CardDrawUtils() 

def draw(image=None, is_cnt=False, cnts=[], coordinates=[]):
    if is_cnt:
        whiteFrame = []
        if type(cnts) != dict:
            whiteFrame = 255 * np.ones(image.shape, np.uint8)
            try:
                whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (232, 54, 0), 3)
            except:
                whiteFrame = cv2.drawContours(whiteFrame, np.expand_dims(np.array(setk[1], dtype=np.int32), axis=0), -1, (232, 54, 0), 3)
        else:
            whiteFrame = 255 * np.ones(image.shape, np.uint8)
            for key, val in cnts.items():
                whiteFrame = cv2.drawContours(whiteFrame, val, -1, (232, 54, 0), 3)
        return whiteFrame
    else:
        for coordinate in coordinates:
            xy = list(map(lambda x: int(x), coordinate)) 
            image = cv2.circle(image, xy, radius=8, color=(232, 54, 39), thickness=5)
        return image

addr = './Datasets/data' 
addr_to_save = './outputs'

img_paths = os.listdir(addr)

for x in img_paths: 
    # x = 'vmax_39.jpg' 
    if '' in x: 
        
        img_path = f'{x}'
        
        img_path = 'gx_16.jpg'  #, vmax_21
        
        # print('Current', img_path)
        
        image = cv2.imread(os.path.join(addr, img_path), 0)
        
        # If the image is potrait rotate it 
        image, potrait_status = rotationUtils.is_potrait_then_rotate(image, rotate_status='START')
        
        # Rotate the image to the center 
        image, central = rotationUtils.image_auto_rotation(image)
         
        # Get the outer line of the card 
        cnts, top, bottom, right, left, \
            one_top, two_bottom, one_right, two_left, \
            top_line, bottom_line, right_line, left_line = lineUtils.get_outermost_line_to_measure(image, central, is_ouside=True, card_type=img_path)
        
        # Draw contour
        image_cnts = draw(image=image, cnts=cnts, is_cnt=True)
        
        # Draw coordinates
        image_with_point = draw(image=image, is_cnt=False, coordinates=[top_line, bottom_line, right_line, left_line])
        
        # Save frame
        rgb = cv2.cvtColor(image_with_point, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)

    44.24 44.34 31.11 31.11
    44.34 44.1  31.26 31.96
    44.49 44.06 31.89 31.61
    44.24 44.38 30.9 32.03
    Final top 44.41 bottom 44.34 right 32.14 left 31.68

# Plot one side 
setk = two_bottom[sorted(two_bottom.keys())[-3]]

image_with_point_ = draw(image=image_cnts, cnts=np.array(setk[1], dtype=np.int32), is_cnt=True)
c = np.array(setk[1], dtype=np.int32)
image_with_point_ = draw(image=image_with_point_, is_cnt=False, coordinates=[[int(np.average(np.squeeze(c)[:, 0])), int(np.average(np.squeeze(c)[:, 1]))]])

# Save frame
rgb = cv2.cvtColor(image_with_point_, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)

 
# Draw contour
image_cnts = draw(image=image, cnts=top, is_cnt=True)
# Save frame
rgb = cv2.cvtColor(image_cnts, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)




# Remove the center of the card to reduce some computation
image_1, removed_b_box = cardRemovalUtils.remove_card_center(image, central)

# Remove some distance from the outer card to prevent of being re-detected
seg_img, new_removed_b_box = cardRemovalUtils.remove_some_slices_from_the_edge(image_1, top_line, bottom_line, right_line, left_line, cut_control=30)
  
# Extract the contour of the inner card 
whiteFrameWithCountour, cnts, gap_removal = contourDetectionUtils.get_contours_inner(seg_img)
       
# Adjust the extracted image into the original image dimension
whiteFrames = cardRemovalUtils.frame_adjustment_to_the_original_size(image_1.shape, whiteFrameWithCountour, removed_b_box, new_removed_b_box, gap_removal)
     
# Extract inner lines of the card 
cnts, top, bottom, right, left, \
    one_top, two_bottom, one_right, two_left, \
    top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate = lineUtils.get_outermost_line_to_measure(whiteFrames, central, is_binary=True, is_ouside=False)
     
# To be used as outer
outer_top_line, outer_bottom_line, outer_right_line, outer_left_line = [top_center_coordinate[0], top_line[1]], [bottom_center_coordinate[0], bottom_line[1]], [right_line[0], right_center_coordinate[1]], [left_line[0], left_center_coordinate[1]]

# IND = -2
# top = np.array(one_top[sorted(one_top.keys())[IND]][1], dtype=np.int32)
# bottom = np.array(two_bottom[sorted(two_bottom.keys())[-3]][1], dtype=np.int32)
# right = np.array(one_right[sorted(one_right.keys())[IND]][1], dtype=np.int32)
# left = np.array(two_left[sorted(two_left.keys())[-5]][1], dtype=np.int32)

# top = top if np.average(np.squeeze(top)[:, 1]) < np.average(np.squeeze(np.array(one_top[sorted(one_top.keys())[-1]][1], dtype=np.int32))[:, 1]) else np.array(one_top[sorted(one_top.keys())[-1]][1], dtype=np.int32)
# bottom = bottom if np.average(np.squeeze(bottom)[:, 1]) > np.average(np.squeeze(np.array(two_bottom[sorted(two_bottom.keys())[-1]][1], dtype=np.int32))[:, 1]) else np.array(two_bottom[sorted(two_bottom.keys())[-1]][1], dtype=np.int32)
# right = right if np.average(np.squeeze(right)[:, 0]) > np.average(np.squeeze(np.array(one_right[sorted(one_right.keys())[-1]][1], dtype=np.int32))[:, 0]) else np.array(one_right[sorted(one_right.keys())[-1]][1], dtype=np.int32)
# left = left if np.average(np.squeeze(top)[:, 0]) < np.average(np.squeeze(np.array(two_left[sorted(two_left.keys())[-1]][1], dtype=np.int32))[:, 0]) else np.array(two_left[sorted(two_left.keys())[-1]][1], dtype=np.int32)
 
whiteFrame = cv2.drawContours(image_cnts, top, -1, (0, 0, 0), 3)
whiteFrame = cv2.drawContours(whiteFrame, bottom, -1, (0, 0, 0), 3)
whiteFrame = cv2.drawContours(whiteFrame, right, -1, (0, 0, 0), 3)
whiteFrame = cv2.drawContours(whiteFrame, left, -1, (0, 0, 0), 3)

im = draw(image=whiteFrame, is_cnt=False, cnts=[], coordinates=[top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate]) 

# Save frame
rgb = cv2.cvtColor(image_with_point_, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)