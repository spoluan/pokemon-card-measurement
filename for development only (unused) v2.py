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

def draw(image=None, is_cnt=False, cnts=[], coordinates=[], thickness=5):
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
            image = cv2.circle(image, xy, radius=8, color=(232, 54, 39), thickness=thickness)
        return image
     

fileUtils = FileUtils()  
rotationUtils = RotationUtils()
contourDetectionUtils = ContourDetectionUtils()
lineUtils = LineUtils()
cardRemovalUtils = CardRemovalUtils()
cardDrawUtils = CardDrawUtils() 

addr = './Datasets/data-fixed-detected' 
addr_to_save = './outputs' 
img_path = 'pokemon_B1.jpg'

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

# Get corners
outer_top_line, outer_bottom_line, outer_right_line, outer_left_line = \
            [top_center_coordinate[0], top_line[1]], \
            [bottom_center_coordinate[0], bottom_line[1]], \
            [right_line[0], right_center_coordinate[1]], \
            [left_line[0], left_center_coordinate[1]]
inner_top_line, inner_bottom_line, inner_right_line, inner_left_line = \
            top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate
_, corners = cardDrawUtils.plot_card_corner_detection(image, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inner_top_line, inner_bottom_line, inner_right_line, inner_left_line)
top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner = corners

get_left_corner = image[top_left_corner[0][1]:top_left_corner[2][1], top_left_corner[0][0]:top_left_corner[1][0]]

# image_with_corner = draw(image=image, is_cnt=False, coordinates=top_left_corner)
# cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_corner) 
cv2.imwrite(os.path.join(addr_to_save, 'left_' + img_path), get_left_corner)  # y, x

# Detect contour corner
# img_cnt, _, x = contourDetectionUtils.get_contours_outer_vbeta(get_left_corner)
# img = 255 * np.ones(get_left_corner.shape, np.uint8)
# img = draw(image=img, is_cnt=False, coordinates=np.squeeze(img_cnt), thickness=1)
# cv2.imwrite(os.path.join(addr_to_save, 'left_cnt_' + img_path), img)
 
# Detect contour corner
dst = cv2.Canny(get_left_corner, 50, 200, None, 3)
# lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10) 
cv2.imwrite(os.path.join(addr_to_save, 'left_cnt_dst_' + img_path), dst)
 

# Filter outer line
print(get_left_corner.shape[0]) 
base_measument = [2, 0] * np.expand_dims(np.arange(0, get_left_corner.shape[0] // 2, 2), axis=0).T 
hold = []
for base in base_measument: 
    for y in range(get_left_corner.shape[1]):
        if dst[base[0]][y] > 0:
            hold.append([base[0], y])
            break
    
# Plot the results
p = 255 * np.ones(get_left_corner.shape, np.uint8)     
for x in hold:
    mage = cv2.circle(p, x, radius=1, color=(232, 54, 39), thickness=1)
cv2.imwrite(os.path.join(addr_to_save, 'left_cnt_dst_mage_res' + img_path), mage)

# Compute curvatory
# Apply code for an example
x = np.array(hold)[:, 0]
y = np.array(hold)[:, 1]
comp_curv = ComputeCurvature()
curvature = comp_curv.fit(x, y)

# Plot the result
theta_fit = np.linspace(-np.pi, np.pi, 180)
x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2)
plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('curvature = {:.3e}'.format(curvature))
plt.show()

# whiteFrame = 255 * np.ones(image.shape, np.uint8)
# for x in sorted(one_right.keys()):
#     whiteFrame = draw(image=whiteFrame, cnts=np.expand_dims(np.array(one_right[x][1], dtype=np.int32), axis=1), is_cnt=True)
# cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrame) 
