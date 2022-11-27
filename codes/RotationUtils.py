# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:42:29 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import numpy as np
import cv2
import imutils
import math
from codes.ContourDetectionUtils import ContourDetectionUtils
from codes.LineUtils import LineUtils

class RotationUtils(object):
    
    def __init__(self):
        self.contourDetectionUtils = ContourDetectionUtils()
        self.lineUtils = LineUtils()
        
    def is_potrait_then_rotate(self, image, rotate_status='START'):
        # Maintain if the image is in a potrait mode rotate it 
        potrait_status = False
        if image.shape[0] < image.shape[1]:
             potrait_status = True
             if rotate_status == 'START':
                 image = imutils.rotate_bound(image, 90) 
             elif rotate_status == 'END':
                 image = imutils.rotate_bound(image, -90) 
        return image, potrait_status
        
    def get_shifting_distance(self, upper_center, lower_center, right_center, left_center):
        # Shifting distance
        x_to_shift = abs(upper_center[0] - lower_center[0])
        y_to_shift = abs(right_center[1] - left_center[1])
        return x_to_shift / 2.0, y_to_shift / 2.0 
     
    def get_rotate_direction(self, upper_center, lower_center):
        rotation_status = 'CENTER'
        
        # Extend the coordinate exceeds the maximum line to the an accurate measurements
        upper_center, lower_center =  self.lineUtils.line_continuation(upper_center, lower_center)
        
        dis = abs(upper_center[0] - lower_center[0])
        if dis >= 5:
            if upper_center[0] > lower_center[0]: # rotate left
                rotation_status = 'ROTATE_LEFT'
            else:
                rotation_status = 'ROTATE_RIGHT' 
                
        return dis, rotation_status
    
    def rotate_image(self, image, rotation_status):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = None
        if rotation_status == 'ROTATE_LEFT':
            M = cv2.getRotationMatrix2D((cX, cY), 0.05, 1.0)
        elif rotation_status == 'ROTATE_RIGHT':
            M = cv2.getRotationMatrix2D((cX, cY), -0.05, 1.0)
        else:
            M = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255)) 
        return rotated

    def rotate_image_vbeta(self, image, val):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = None 
        M = cv2.getRotationMatrix2D((cX, cY), val, 1) 
        rotated = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)) 
        return rotated 

    def get_degree(self, hold):   
        val_a, val_b, val_a2, val_b2 = hold
        opposite = val_a - val_a2
        adjacent = val_b - val_b2
        hypotenuse = math.sqrt(pow(opposite, 2) + pow(adjacent, 2)) 
        degrees = math.asin(opposite / hypotenuse) * 180 / math.pi 
        return degrees 
    
    def image_auto_rotation(self, image):
          
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        contours_, hierarchy = self.contourDetectionUtils.get_contours_outer(image) 
        cnts, top_left, top_right, bottom_left, bottom_right = self.lineUtils.get_conf_outer_line(contours_)
        upper_center, lower_center, right_center, left_center = self.lineUtils.get_center_line_on_image(top_left, top_right, bottom_left, bottom_right)
        dis, rotation_status = self.get_rotate_direction(upper_center, lower_center)
        
        central = [np.average([upper_center[0], lower_center[0], right_center[0], left_center[0]]), np.average([upper_center[1], lower_center[1], right_center[1], left_center[1]])]
        
        count = 0
        min_dis = 99999
        while rotation_status != 'CENTER':
            contours_, hierarchy = self.contourDetectionUtils.get_contours_outer(image)
            cnts, top_left, top_right, bottom_left, bottom_right = self.lineUtils.get_conf_outer_line(contours_)
            upper_center, lower_center, right_center, left_center = self.lineUtils.get_center_line_on_image(top_left, top_right, bottom_left, bottom_right)
            
            dis, rotation_status = self.get_rotate_direction(upper_center, lower_center)
            image = self.rotate_image(image, rotation_status)
            
            if dis < min_dis:
                min_dis = dis 
            else:
                count += 1
                
            print(dis, rotation_status, count)
            
            if dis < 5:
                break

            if count > 5 and dis < 10: 
                break 
            
            if count > 20 and dis < 15: 
                break 
            
            if count > 20 and dis < 20: 
                break 
            
            # Will automatically go out when there is no more update
            if count > 30 and dis < 50: 
                break    
        
        # try:
        #     angle_pre = self.get_angle(image)
        #     image = self.rotate_image_vbeta(image, angle_pre)
        #     angle_post = self.get_angle(image)
        #     print('Current results', dis, angle_pre, angle_post)
        # except:
        #     pass
        
        print('Done rotating')  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return image, central

    def draw_contours(self, image=None, cnts=[]):  
        image_cnts = 255 * np.ones(image.shape, np.uint8) 
        image_cnts = cv2.drawContours(image_cnts, cnts, -1, (0, 0, 0), -1)
        # image_cnts = cv2.drawContours(image_cnts, cnts, -1, (232, 54, 0), 3)
        image_cnts = cv2.cvtColor(image_cnts, cv2.COLOR_BGR2GRAY)
        return image_cnts 

    def get_centroid(self, linesP):
        aligned = []
        for xx in linesP:
            x = np.squeeze(xx)
            aligned.append([x[0], x[1]])
            aligned.append([x[2], x[3]])
        x = [p[0] for p in aligned]
        y = [p[1] for p in aligned]
        centroid = (sum(x) / len(aligned), sum(y) / len(aligned))
        return centroid

    def get_angle(self, image): 
        # Detect outer contour 
        cnts, hierarchy = self.contourDetectionUtils.get_contours_outer_vbeta(image) # get_contour_full_border(image) 
         
        # Draw contour
        image_cnts = self.draw_contours(image=image, cnts=cnts)
         
        # Line detection
        linesP = cv2.HoughLinesP(255 - image_cnts, 1, np.pi / 180, 210, None, 100, 500) # -1: length minimum -2: line ratio
         
        # Get controid
        central = self.get_centroid(linesP)
        
        # Get the longest line
        max_line = 0 
        hold = [] 
        for i in range(0, len(linesP)):
            lin = linesP[i][0] 
            if abs(lin[0] - lin[2]) < 70 and (lin[0] < central[0] or lin[0] > central[0]):
                dist = np.linalg.norm(np.array([lin[0], lin[1]]) - np.array([lin[2], lin[3]]))
                if dist > max_line:
                    max_line = dist 
                    hold = lin
        
        # Get the angle line 
        deg = self.get_degree(hold)
        # if self.get_degree_double_check(hold) != 0:
        #     deg = self.get_degree(hold)
        return deg
    
    # Un-used method
    def get_img_center_coordinates(self, image, upper_center, lower_center, right_center, left_center, x_to_shift, y_to_shift):
        
        # Get the farest coordinates from the outest of the original image size 
        start_upper_left, start_upper_right, start_lower_left, start_lower_right = [0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]] 
        if upper_center[0] > lower_center[0]: # Shift right
            start_upper_left = [int(start_upper_left[0] + x_to_shift), start_upper_left[1]] 
        elif upper_center[0] < lower_center[0]:
            start_upper_right = [int(start_upper_right[0] - x_to_shift), start_upper_right[1]] 
            
        if left_center[1] > right_center[1]:
            start_upper_left = [start_upper_left[0], int(start_upper_left[1] + y_to_shift)]
        elif left_center[1] < right_center[1]:
            start_upper_right = [start_upper_right[0], start_upper_right[1] + y_to_shift]
        
        return start_upper_left, start_upper_right, start_lower_left, start_lower_right