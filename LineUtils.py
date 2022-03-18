# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:57:32 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import cv2
import numpy as np
import pandas as pd
import math
from ContourDetectionUtils import ContourDetectionUtils

class LineUtils(object):
    
    def __init__(self):
        self.contourDetectionUtils = ContourDetectionUtils()
    
    def centroid(self, vertexes):
         _x_list = [vertex [0] for vertex in vertexes]
         _y_list = [vertex [1] for vertex in vertexes]
         _len = len(vertexes)
         _x = float(float(sum(_x_list)) / float(_len))
         _y = float(float(sum(_y_list)) / float(_len))
         return [_x, _y] 
    
    def get_center_line_on_image(self, top_left, top_right, bottom_left, bottom_right):
        upper_center, lower_center, right_center, left_center = [], [], [], []
        upper_center = self.centroid([top_right, top_left])
        lower_center = self.centroid([bottom_right, bottom_left])
        right_center = self.centroid([top_right, bottom_right])
        left_center = self.centroid([bottom_left, top_left])
        return upper_center, lower_center, right_center, left_center 
    
    def check_distance_to_clasify_xy(self, z, val, idx=1): # 1 is for accessing y axis
    
        # Define the line range
        ranges = 2 
        val = val[idx]  
        try:
            keys = np.array(list(z.keys()))
            keys_to_check = keys[np.where((keys >= val - ranges) & (keys <= val + ranges))] 
            if len(keys_to_check) > 0: 
                return keys_to_check[0]
            return None
        except:    
            return None
    
    def get_horizontal_and_vertical_lines(self, cnts, idx=1, img_dim=[]):  
        print('Ger horizontal and vertical lines')
        # Allocate all the cnts into one array
        new_cnts = []
        for x in cnts:
            new_cnts.extend(x)
            
        # Sort values 
        cnts_pd = pd.DataFrame(np.squeeze(new_cnts))
        
        # Sort by y if idx == 1
        if idx == 1:
            cnts_pd = cnts_pd.sort_values(1)  
        
        # Provide a gap as for the starting point to start the line detection 
        gap = 20
          
        z = {}
        skip = 2
        skip_loop = skip
        for y in cnts_pd.itertuples(index=False):  
            if skip_loop == 0:
                skip_loop = 4 
                # Check the value to prevent from getting the border aligned with the image dimension
                if (y[0] > 0 + gap and y[0] < img_dim[1] - gap) and \
                    (y[1] > 0 + gap and y[1] < img_dim[0] - gap):
                        
                    # To check in which class it will be classified
                    check = self.check_distance_to_clasify_xy(z, np.array(y), idx=idx)  # 1 is for accessing y axis
                    
                    if check is None: 
                        # Use the y value as key to store all values within its range of lines
                        if y[idx] not in z:
                            z[y[idx]] = []
                        z[y[idx]].append(list(y))
                    else:
                        z[check].append(list(y))
            skip_loop -= 1 
         
        _lines = z.copy() 
        for k, v in z.items(): 
            if len(v) < 100:  
                _lines.pop(k)  
                
        return z, _lines   
    
    def get_coordinate_to_measure(self, _lines, central, status='HORIZONTAL_MEASUREMENT'):
        
        # Config for horizontal and vertical measurement
        idx = 1
        key_1 = 'TOP'
        key_2 = 'BOTTOM'  
        if status == 'VERTICAL_MEASUREMENT':
            idx = 0
            key_1 = 'LEFT'
            key_2 = 'RIGHT'
        
        # Get the average of the detected lines and make the array length as the dict key
        _lines_cp = {key_1: {}, key_2: {}}
        for x, y in _lines.items():
            val = [np.average(np.squeeze(y)[:, 0]), np.average(np.squeeze(y)[:, 1])]
            if val[idx] > central[idx]:
                _lines_cp[key_2][len(np.squeeze(y)[:, idx])] = [val, y]
            else:
                _lines_cp[key_1][len(np.squeeze(y)[:, idx])] = [val, y]
           
        one, two = _lines_cp[key_1], _lines_cp[key_2]
        
        print(f'Get {status} detected ...')
        return one, two
    
    def line_continuation(self, p1, p2):
        
        line_length = 500
        
        theta = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        
        endpt_x = int(p1[0] + line_length * np.cos(theta))
        endpt_y = int(p1[1] + line_length * np.sin(theta))
        
        endpt_x1 = int(p1[0] - line_length * np.cos(theta))
        endpt_y1 = int(p1[1] - line_length * np.sin(theta))
         
        return (endpt_x, endpt_y), (endpt_x1, endpt_y1) 
    
    def get_conf_outer_line(self, contours_):  
        
        # Get the outest rectangle coordinates
        perimeter, approximated_shape, cnts = 0, [], [] 
        t_per = 0
        for x in contours_:
            try: 
                contour = x # Take the inner line from the outermost line
                perimeter = cv2.arcLength(contour, True)   
                if t_per < perimeter:
                    t_per = max(t_per, perimeter)
                    cnts = x 
                    approximated_shape = np.squeeze(cv2.approxPolyDP(cnts, 0.02 * perimeter, True))
            except:
                pass  
        
        # Classify the coordinates
        top_left, top_right, bottom_left, bottom_right = [], [], [], []
        if len(approximated_shape) == 4:
            center = self.centroid(approximated_shape) 
            for x in approximated_shape:
                if x[0] < center[0] and x[1] < center[1]:
                    top_left = x
                if x[0] > center[0] and x[1] < center[1]:
                    top_right = x
                if x[0] < center[0] and x[1] > center[1]:
                    bottom_left = x
                if x[0] > center[0] and x[1] > center[1]:
                    bottom_right = x
                
        return cnts, top_left, top_right, bottom_left, bottom_right 
     
    def get_outermost_line_to_measure(self, image, central, is_binary=False, is_ouside=True, card_type='pokemon'):
        print('Extract the outer line of the card')
        img_dim = image.shape
        
        if not is_binary:
            cnts, hierarchy = self.contourDetectionUtils.get_contour_full_border(image)
        else:
            print('Extract the inner lines of the card')
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
            cnts, hierarchy = cv2.findContours(gray.copy(), 
                                               cv2.RETR_LIST ,
                                               cv2.CHAIN_APPROX_NONE)   
         
        # Get horizontal line of the card border contour
        all_hor, hor_lines = self.get_horizontal_and_vertical_lines(cnts, idx=1, img_dim=img_dim) # 1 indicates y axis
        one_top, two_bottom = self.get_coordinate_to_measure(hor_lines, central, status='HORIZONTAL_MEASUREMENT')
        
        # Get vertical line of the card border contour
        all_ver, ver_lines = self.get_horizontal_and_vertical_lines(cnts, idx=0, img_dim=img_dim) # 0 indicates x axis
        two_left, one_right = self.get_coordinate_to_measure(ver_lines, central, status='VERTICAL_MEASUREMENT')
         
        print('Get the highest number of points')
        IND = -1 #  Default
        INDS = np.arange(-150, 0, 1)[::-1]
        top = np.array(one_top[sorted(one_top.keys())[IND]][1], dtype=np.int32)
        bottom = np.array(two_bottom[sorted(two_bottom.keys())[IND]][1], dtype=np.int32)
        right = np.array(one_right[sorted(one_right.keys())[IND]][1], dtype=np.int32)
        left = np.array(two_left[sorted(two_left.keys())[IND]][1], dtype=np.int32) 
        
        print('Length', len(sorted(one_top.keys())), len(sorted(two_bottom.keys())), len(sorted(one_right.keys())), len(sorted(two_left.keys())))
         
        def g_avg(val, idx=0):
            return np.average(np.squeeze(val)[:, idx])
        
        
        def get_mm(center, point2, top_bottom=True):
            if top_bottom:
                pixels = math.sqrt(abs(int(center[0]) - int(center[0]))**2 + abs(int(center[1]) - int(point2[1]))**2)
                mm = round((pixels * 25.4) / 720, 2)
            else:
                pixels = math.sqrt(abs(int(center[0]) - int(point2[0]))**2 + abs(int(center[1]) - int(center[1]))**2)
                mm = round((pixels * 25.4) / 720, 2)
            return mm
        
        def get_value(array, keys, array_max, c_array, threshold, c_array_max, top_bottom=True, show_log=False): 
            for x in range(len(sorted(keys.keys()))): 
                t_array = np.array(keys[sorted(keys.keys())[x]][1], dtype=np.int32)  
                array_ = [int(np.average(np.squeeze(t_array)[:, 0])), int(np.average(np.squeeze(t_array)[:, 1]))]
                array_to_center = get_mm(central, array_, top_bottom=top_bottom) 
                if array_to_center >= threshold[0] and array_to_center <= threshold[1]:
                    # if top_to_center > c_top:
                    array = t_array 
                    c_array = array_to_center
                    
                # Maintain the max value
                if c_array_max < array_to_center:
                    c_array_max = array_to_center
                    array_max = t_array
                if show_log:
                    print(array_to_center)
            return array, c_array, c_array_max, array_max
                    
        top_ = [int(np.average(np.squeeze(top)[:, 0])), int(np.average(np.squeeze(top)[:, 1]))]
        bottom_ = [int(np.average(np.squeeze(bottom)[:, 0])), int(np.average(np.squeeze(bottom)[:, 1]))]
        right_ = [int(np.average(np.squeeze(right)[:, 0])), int(np.average(np.squeeze(right)[:, 1]))]
        left_ = [int(np.average(np.squeeze(left)[:, 0])), int(np.average(np.squeeze(left)[:, 1]))]
         
        top_to_center = get_mm(central, top_, top_bottom=True)
        bottom_to_center = get_mm(central, bottom_, top_bottom=True)
        right_to_center = get_mm(central, right_, top_bottom=False)
        left_to_center = get_mm(central, left_, top_bottom=False)
        
        print('start', top_to_center, bottom_to_center, right_to_center, left_to_center)
         
        # Detect the outest for inner line detection
        try:
            threshold = 120
            
            if 'pokemon_' in card_type.lower():
                print('POKEMON')
                normal_right_left_distance_to_center = [30.8, 32.0]
                normal_top_bottom_distance_to_center = [44.10, 44.3] 
            elif 'vmax_' in card_type.lower():
                print('VMAX')
                normal_right_left_distance_to_center = [31.60, 32.4] 
                normal_top_bottom_distance_to_center = [44.3, 44.28] 
            elif 'gx_' in card_type.lower():
                print('GX')
                normal_right_left_distance_to_center = [32.3, 32.4]
                normal_top_bottom_distance_to_center = [44.3, 44.2]
            elif 'vcard_' in card_type.lower():
                print('VCARD')
                normal_right_left_distance_to_center = [31.58, 32.0] 
                normal_top_bottom_distance_to_center = [40.42, 44.45]  
            elif 'vstar_' in card_type.lower():
                print('VSTAR')
                normal_right_left_distance_to_center = [31.58, 32.0] 
                normal_top_bottom_distance_to_center = [40.42, 44.23] 
            elif 'trainers_' in card_type.lower():
                print('TRAINERS')
                normal_right_left_distance_to_center = [32.3, 32.5] # Fixed
                normal_top_bottom_distance_to_center = [44.3, 44.5] 
            else:
                print('DEFAULT')
                normal_right_left_distance_to_center = [32.3, 32.5] 
                normal_top_bottom_distance_to_center = [44.3, 44.5] 
            
            c_top = 0
            c_bottom = 0
            c_right = 0
            c_left = 0
            
            # Maintain max_value
            c_top_max, top_max = 0, []
            c_bottom_max, bottom_max = 0, []
            c_right_max, right_max = 0, []
            c_left_max, left_max = 0, []
            
            if not is_ouside:
                for x in INDS:
                    t_top = np.array(one_top[sorted(one_top.keys())[x]][1], dtype=np.int32)
                    t_bottom = np.array(two_bottom[sorted(two_bottom.keys())[x]][1], dtype=np.int32)
                    t_right = np.array(one_right[sorted(one_right.keys())[x]][1], dtype=np.int32)
                    t_left = np.array(two_left[sorted(two_left.keys())[x]][1], dtype=np.int32)
                    
                    top = top if g_avg(top, 1) < g_avg(t_top, 1) or abs(len(top) - len(t_top)) >= threshold else t_top
                    bottom = bottom if g_avg(bottom, 1) > g_avg(t_bottom, 1) or abs(len(bottom) - len(t_bottom)) >= threshold else t_bottom
                    right = right if g_avg(right, 0) > g_avg(t_right, 0) or abs(len(right) - len(t_right)) >= threshold else t_right
                    left = left if g_avg(left, 0) < g_avg(t_left, 0) or abs(len(left) - len(t_left)) >= threshold else t_left 
            else:
                top, c_top, c_top_max, top_max = get_value(top, one_top, top_max, c_top, normal_top_bottom_distance_to_center, c_top_max, top_bottom=True, show_log=False)
                bottom, c_bottom, c_bottom_max, bottom_max = get_value(bottom, two_bottom, bottom_max, c_bottom, normal_top_bottom_distance_to_center, c_bottom_max, top_bottom=True, show_log=False)
                right, c_right, c_right_max, right_max = get_value(right, one_right, right_max, c_right, normal_right_left_distance_to_center, c_right_max, top_bottom=False, show_log=False)
                left, c_left, c_left_max, left_max = get_value(left, two_left, left_max, c_left, normal_right_left_distance_to_center, c_left_max, top_bottom=False)
       
        except Exception as err:
            print('Passed!', err) 
        
        # If there is no update assign with the maximum value
        if is_ouside:
            if c_top == 0:
                c_top = c_top_max
                top = top_max
            if c_bottom == 0:
                c_bottom = c_bottom_max
                bottom = bottom_max
            if c_right == 0:
                c_right = c_right_max
                right = right_max
            if c_left == 0:
                c_left = c_left_max
                left = left_max
            
        print('Final', 'top', c_top, 'bottom', c_bottom, 'right', c_right, 'left', c_left)
        
        print('Get the center of the line')
        top_center_coordinate = [int(np.average(np.squeeze(top)[:, 0])), int(np.average(np.squeeze(top)[:, 1]))]
        bottom_center_coordinate = [int(np.average(np.squeeze(bottom)[:, 0])), int(np.average(np.squeeze(bottom)[:, 1]))]
        right_center_coordinate = [int(np.average(np.squeeze(right)[:, 0])), int(np.average(np.squeeze(right)[:, 1]))]
        left_center_coordinate = [int(np.average(np.squeeze(left)[:, 0])), int(np.average(np.squeeze(left)[:, 1]))]
        
        return cnts, top, bottom, right, left, \
            one_top, two_bottom, one_right, two_left, \
                top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate
    
    
