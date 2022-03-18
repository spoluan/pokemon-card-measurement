# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:42:29 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import numpy as np
import cv2
import imutils
from ContourDetectionUtils import ContourDetectionUtils
from LineUtils import LineUtils

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
            M = cv2.getRotationMatrix2D((cX, cY), 0.08, 1.0)
        elif rotation_status == 'ROTATE_RIGHT':
            M = cv2.getRotationMatrix2D((cX, cY), -0.08, 1.0)
        else:
            M = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255)) 
        return rotated
    
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
            
            if count > 5 and dis < 10: 
                break 
            
            if count > 20 and dis < 15: 
                break 
            
            if count > 20 and dis < 20: 
                break 
            
            # Will automatically go out when there is no more update
            if count > 30 and dis < 50: 
                break    
            
        print('Done rotating')    
        return image, central
    
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