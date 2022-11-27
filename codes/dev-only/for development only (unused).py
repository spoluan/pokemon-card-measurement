# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:23:53 2022

@author: Sevendi Eldrige Rifki Poluan
     # This file is used for development only
     
     # Ref
         https://github.com/alkasm/magicwand
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
     
def get_degree(hold):  
    return math.atan2(hold[1] - hold[3], hold[0] - hold[2])

def get_length(hold):  
    val_a, val_b, val_a2, val_b2 = hold
    opposite = val_a - val_a2
    adjacent = val_b - val_b2
    hypotenuse = math.sqrt(pow(opposite, 2) + pow(adjacent, 2)) 
    degrees = math.asin(opposite / hypotenuse) * 180 / math.pi 
    return degrees

def rotate_image(image, val, absolute=False):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = None
    rotate = 0.1
    if absolute:
        M = cv2.getRotationMatrix2D((cX, cY), val, 1)
    else:
        M = cv2.getRotationMatrix2D((cX, cY), rotate if val < 0 else -rotate, 1)
    rotated = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)) 
    return rotated 

def get_centroid(linesP):
    aligned = []
    for xx in linesP:
        x = np.squeeze(xx)
        aligned.append([x[0], x[1]])
        aligned.append([x[2], x[3]])
    x = [p[0] for p in aligned]
    y = [p[1] for p in aligned]
    centroid = (sum(x) / len(aligned), sum(y) / len(aligned))
    return centroid

def get_angle(image):
     
    cnts, hierarchy = contourDetectionUtils.get_contours_outer(image)
    # Draw contour
    image_cnts = draw(image=image, cnts=cnts, is_cnt=True)
     
    gray = cv2.cvtColor(image_cnts, cv2.COLOR_BGR2GRAY)
    linesP = cv2.HoughLinesP(255 - gray, 1, np.pi / 180, 180, None, 100, 200) # -1: length minimum -2: line ratio
    
    centroid = get_centroid(linesP)
    
    # Save frame 
    # cv2.imwrite(os.path.join(addr_to_save, img_path), image_cnts)
        
    # rotated_img = rotate_image(image, '') 
    # rotated_img = rotate_image(rotated_img, '')
    # deg = get_degree(rotated_img)   
    
    # Draw contour
    # image_cnts = draw(image=image, cnts=sel_, is_cnt=True) 
    # image_with_point = draw(image=image_cnts, is_cnt=False, coordinates=[first_point, second_point]) 
    
    maxx = 0
    whiteFrame = 255 * np.ones(image.shape, np.uint8) 
    hold = []
    de = []
    for i in range(0, len(linesP)):
        lin = linesP[i][0]
        if abs(lin[0] - lin[2]) < 70 and (lin[0] < centroid[0] or lin[0] > centroid[0]):
            dist = np.linalg.norm(np.array([lin[0], lin[1]]) - np.array([lin[2], lin[3]]))
            if dist > maxx:
                maxx = dist 
                hold = lin
            cv2.line(whiteFrame, (lin[0], lin[1]), (lin[2], lin[3]), (0,0,255), 3, 4)
            # de.append([dist, lin])
             
    
    # hold = de[2][1]
    whiteFrame = 255 * np.ones(image.shape, np.uint8)
    whiteFrame = cv2.line(whiteFrame, (hold[0], hold[1]), (hold[2], hold[3]), (0, 0, 255), 3, 4)
    # whiteFrame = cv2.circle(whiteFrame, [int(centroid[0]), int(centroid[1])], radius=8, color=(232, 54, 39), thickness=5)
    
    # # Save frame 
    cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrame)
     
    # hold = np.array([ 339, 3224, 287, 270], dtype=np.int32) # 1.0084877614404757
    # hold = np.array([ 287, 3224, 339, 270], dtype=np.int32) # -1.0084877614404757
    # hold = np.array([1846,  148, 1885, 2387], dtype=np.int32)
     
    deg = get_length(hold)
    print(deg)
        
    return deg, [hold[0], hold[2]]


def sobel_algorithm(matrix): 
    
    dilate_img = cv2.dilate(np.array(matrix), np.ones([3, 3], dtype=np.uint8), iterations=1)
    
    blur = cv2.GaussianBlur(dilate_img, (5, 5), 0)

    # Apply Sobelx in high output datatype 'float32'
    # and then converting back to 8-bit to prevent overflow
    sobelx_64 = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    absx_64 = np.absolute(sobelx_64)
    sobelx_8u1 = absx_64 / absx_64.max() * 255
    sobelx_8u = np.uint8(sobelx_8u1)
     
    # Similarly for Sobely
    sobely_64 = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    absy_64 = np.absolute(sobely_64)
    sobely_8u1 = absy_64 / absy_64.max() * 255
    sobely_8u = np.uint8(sobely_8u1)
     
    # From gradients calculate the magnitude and changing
    # it to 8-bit (Optional)
    mag = np.hypot(sobelx_8u, sobely_8u)
    mag = mag / mag.max() * 255
    mag = np.uint8(mag)
    
    C = cv2.Canny(np.array(mag), 0, 180)
    _, thresh = cv2.threshold(C, 35, 255, cv2.THRESH_BINARY_INV)
    
    # _, thresh = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY_INV)
    # gray = cv2.cvtColor(np.array(thresh), cv2.COLOR_RGB2GRAY) 
    
    # rgb = cv2.cvtColor(whiteFrame, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)
    
    return thresh

addr = './Datasets/data' 
addr_to_save = './outputs'

img_paths = os.listdir(addr)

for x in img_paths: 
    # x = 'vmax_39.jpg' 
    if '' in x: 
        
        # img_path = f'{x}'
        
        img_path = 'vmax_21.jpg'  #, gx_10, gx_47
        
        # print('Current', img_path)
        
        image = cv2.imread(os.path.join(addr, img_path), 1)
        
        # If the image is potrait rotate it 
        image, potrait_status = rotationUtils.is_potrait_then_rotate(image, rotate_status='START')
         
        # image = rotationUtils.image_auto_rotation(image)
         
        # Save frame
        # rgb = cv2.cvtColor(threshed, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)
         
        
        # Rotate the image to the center 
        image, central = rotationUtils.image_auto_rotation(image)
         
        
        
        # deg = get_length(hold)
        
        # print(deg)
        
        # image = cv2.imread(os.path.join(addr, img_path), 1)
        
        # angle, hold = get_angle(image)
        # print(angle)
        
        # image = rotate_image(image, angle, absolute=True)
        
        cv2.imwrite(os.path.join(addr_to_save, img_path), image)
        
        # angle, hold = get_angle(image)
        # print(angle)
        # img = image.copy()
        # if hold[0] > hold[1]:
        #     image = rotate_image(img, angle)
        # elif hold[0] < hold[1]:
        #     image = rotate_image(image, -angle)
        # cv2.imwrite(os.path.join(addr_to_save, img_path), image)
        
         
        
        
        
        
        
        # image_with_point = draw(image=whiteFrame, is_cnt=False, coordinates=[[hold1[0], hold1[1]], [hold1[2], hold1[3]], [hold2[0], hold2[1]], [hold2[2], hold2[3]]])

        # image_with_point = draw(image=whiteFrame, is_cnt=False, coordinates=[cen])
         
        
        
        # Get the outer line of the card 
        cnts, top, bottom, right, left, \
            one_top, two_bottom, one_right, two_left, \
            top_line, bottom_line, right_line, left_line, central = lineUtils.get_outermost_line_to_measure(image, central, is_ouside=True, card_type=img_path)
        
         
        # Draw contour
        # image_cnts = draw(image=image, cnts=cnts, is_cnt=True)
        
        # Draw coordinates
        
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        image_with_point = draw(image=rgb, is_cnt=False, coordinates=[top_line, bottom_line, right_line, left_line, central])
        cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_point)
        
        # Plot one side 
        # setk = two_bottom[sorted(two_bottom.keys())[-3]]

        # image_with_point_ = draw(image=image_cnts, cnts=np.array(setk[1], dtype=np.int32), is_cnt=True)
        # c = np.array(setk[1], dtype=np.int32)
        # image_with_point_ = draw(image=image_with_point_, is_cnt=False, coordinates=[[int(np.average(np.squeeze(c)[:, 0])), int(np.average(np.squeeze(c)[:, 1]))]])

        # # Save frame 
        cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_point)

 
        # Draw contour
        # image_cnts = draw(image=image, cnts=top, is_cnt=True)
        # Save frame
        # rgb = cv2.cvtColor(image_cnts, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(addr_to_save, img_path), rgb) 
        
        # Remove the center of the card to reduce some computation
        image_1, removed_b_box = cardRemovalUtils.remove_card_center(image, central)
        
        # Save frame 
        cv2.imwrite(os.path.join(addr_to_save, img_path), image_1)
        
        # Remove some distance from the outer card to prevent of being re-detected
        seg_img, new_removed_b_box = cardRemovalUtils.remove_some_slices_from_the_edge(image_1, top_line, bottom_line, right_line, left_line, cut_control=20)
        
        # Save frame 
        cv2.imwrite(os.path.join(addr_to_save, img_path), seg_img)
         
        # cc = sobel_algorithm(seg_img)
         
        # Extract the contour of the inner card 
        whiteFrameWithCountour, cnts, gap_removal = contourDetectionUtils.get_contours_inner(seg_img)
        
        # Save frame 
        cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrameWithCountour) 
               
        # Adjust the extracted image into the original image dimension
        whiteFrames = cardRemovalUtils.frame_adjustment_to_the_original_size(image_1.shape, whiteFrameWithCountour, removed_b_box, new_removed_b_box, gap_removal)
        
        # Save frame 
        cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrames) 
             
        cnts, hierarchy = contourDetectionUtils.get_contours_outer_vbeta(whiteFrames)
        all_ver, ver_lines = lineUtils.get_horizontal_and_vertical_lines(cnts, idx=0, img_dim=whiteFrames.shape) # 0 indicates x axis
        two_left, one_right = lineUtils.get_coordinate_to_measure(ver_lines, central, status='VERTICAL_MEASUREMENT')
        
        
        whiteFrame = 255 * np.ones(image.shape, np.uint8)
        for x in sorted(one_right.keys()):
            whiteFrame = draw(image=whiteFrame, cnts=np.expand_dims(np.array(one_right[x][1], dtype=np.int32), axis=1), is_cnt=True)
            
        
        
        # Extract inner lines of the card 
        cnts, top, bottom, right, left, \
            one_top, two_bottom, one_right, two_left, \
            top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate = lineUtils.get_outermost_line_to_measure(whiteFrames, central, is_binary=True, is_ouside=False, card_type=img_path)
        
            
            # # Plot one side 
            setk = two_bottom[sorted(one_right.keys())[-1]]
    
            # whiteFrame = 255 * np.ones(image.shape, np.uint8)
            # image_with_point_ = draw(image=whiteFrame, cnts=cnts, is_cnt=True)
            # c = np.array(setk[1], dtype=np.int32) 
            
            # image_with_point_ = draw(image=image, is_cnt=False, coordinates=[[int(np.average(np.squeeze(c)[:, 0])), int(np.average(np.squeeze(c)[:, 1]))]])
    
            # # Save frame 
            # cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_point_)
        
            # # Draw contour
            # image_cnts = draw(image=image, cnts=cnts, is_cnt=True)
        
        # Draw coordinates
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        image_with_point = draw(image=rgb, is_cnt=False, coordinates=[top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate, central])
             
        # Save frame 
        cv2.imwrite(os.path.join(addr_to_save, img_path), image_with_point) 
        
           
        
        inner_top_line, inner_bottom_line, inner_right_line, inner_left_line = \
            top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate
         
        im = draw(image=image_with_point, is_cnt=False, cnts=[], coordinates=[inner_top_line, inner_bottom_line, inner_right_line, inner_left_line]) 
    
        # Save frame
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)