# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:29:59 2022

@author: Sevendi Eldrige Rifki Poluan
"""
 
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt    
import math
 
class CardMeasurement(object):
    
    def __init__(self):
        pass
    
    def centroid(self, vertexes):
         _x_list = [vertex [0] for vertex in vertexes]
         _y_list = [vertex [1] for vertex in vertexes]
         _len = len(vertexes)
         _x = float(float(sum(_x_list)) / float(_len))
         _y = float(float(sum(_y_list)) / float(_len))
         return [_x, _y]
    
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
    
    def get_image_center(self, top_left, top_right, bottom_left, bottom_right):
        upper_center, lower_center, right_center, left_center = [], [], [], []
        upper_center = self.centroid([top_right, top_left])
        lower_center = self.centroid([bottom_right, bottom_left])
        right_center = self.centroid([top_right, bottom_right])
        left_center = self.centroid([bottom_left, top_left])
        return upper_center, lower_center, right_center, left_center
         
    def get_contours_outer(self, image):
        # Convert to grayscale, and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
     
        edged = cv2.dilate(cl1, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        _, edged = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)
     
        # Find contours in the edge map
        cnts, hierarchy = cv2.findContours(edged.copy(), 
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
       
        # Sort the countour based on their area
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
        
        return cnts, hierarchy
    
    # Use for extracting the coordinate of the outermost
    def get_contour_full_border(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        edged = cv2.dilate(rgb, np.ones((7,7), np.uint8)) 
        edged = cv2.medianBlur(edged, 21) 
        
        # Remove any card shadow
        _, edged = cv2.threshold(edged, 128, 255, cv2.THRESH_BINARY)
        
        # Improving the detection performance
        _, edged = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)
        
        gray = cv2.cvtColor(edged, cv2.COLOR_RGB2GRAY)
        _, edged = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
        _, edged= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
        edged = cv2.filter2D(edged, -1, kernel_sharpening)  
        edged = cv2.Canny(edged, 30, 100) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel) 
        th, threshed = cv2.threshold(morphed, 100, 255, cv2.THRESH_OTSU) 
        edged = cv2.dilate(threshed, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1) 
        cnts, hierarchy = cv2.findContours(edged.copy(), 
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True) 
        return cnts, hierarchy
    
    # Use for extracting the inner border line (option two)
    def get_countour_shadow_card(self, seg_img): 
        rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)  
        edged = cv2.dilate(rgb, np.ones((7,7), np.uint8)) 
        edged = cv2.medianBlur(edged, 21)
        edged = 255 - cv2.absdiff(seg_img, edged)
        edged = cv2.normalize(edged, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      
        # Improving the detection performance
        _, edged = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)
        
        gray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)  
        th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU) 
        cnts, hierarchy = cv2.findContours(gray.copy(), 
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE) 
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True) 
        whiteFrame = 255 * np.ones(seg_img.shape, np.uint8)
        whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), 3)
        plt.imshow(whiteFrame) 
        
        return cnts
     
    # Use for extracting the inner border line (option one)
    def get_contours_all(self, seg_img): 
        rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)  
        edged = cv2.dilate(rgb, np.ones((7,7), np.uint8)) 
        edged = cv2.medianBlur(edged, 21) 
        
        # Remove any card shadow
        _, edged = cv2.threshold(edged, 128, 255, cv2.THRESH_BINARY)
      
        gray = cv2.cvtColor(edged, cv2.COLOR_RGB2GRAY) 
        th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
        cnts, hierarchy = cv2.findContours(threshed.copy(), 
                                           cv2.RETR_CCOMP ,
                                           cv2.CHAIN_APPROX_NONE) 
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True) 
        return cnts
     
    def get_shifting_distance(self, upper_center, lower_center, right_center, left_center):
        # Shifting distance
        x_to_shift = abs(upper_center[0] - lower_center[0])
        y_to_shift = abs(right_center[1] - left_center[1])
        return x_to_shift / 2.0, y_to_shift / 2.0
    
    def get_img_center_coordinates(self, upper_center, lower_center, right_center, left_center, x_to_shift, y_to_shift):
        
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
    
    def get_rotate_direction(self, upper_center, lower_center):
        rotation_status = 'CENTER'
        
        dis = abs(upper_center[0] - lower_center[0])
        if dis >= 20:
            if upper_center[0] > lower_center[0]: # rotate left
                rotation_status = 'ROTATE_LEFT'
            else:
                rotation_status = 'ROTATE_RIGHT'
        print(rotation_status, upper_center, lower_center, abs(upper_center[0] - lower_center[0]))
        return dis, rotation_status
         
    def rotate_image(self, image, rotation_status):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = None
        if rotation_status == 'ROTATE_LEFT':
            M = cv2.getRotationMatrix2D((cX, cY), 0.1, 1.0)
        elif rotation_status == 'ROTATE_RIGHT':
            M = cv2.getRotationMatrix2D((cX, cY), -0.1, 1.0)
        else:
            M = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255,255,255)) 
        return rotated
    
    def image_center(self, image):
          
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        contours_, hierarchy = self.get_contours_outer(image)
        cnts, top_left, top_right, bottom_left, bottom_right = self.get_conf_outer_line(contours_)
        upper_center, lower_center, right_center, left_center = self.get_image_center(top_left, top_right, bottom_left, bottom_right)
        dis, rotation_status = self.get_rotate_direction(upper_center, lower_center)
        
        central = [np.average([upper_center[0], lower_center[0], right_center[0], left_center[0]]), np.average([upper_center[1], lower_center[1], right_center[1], left_center[1]])]
        
        count = 0
        min_dis = 99999
        while rotation_status != 'CENTER':
            contours_, hierarchy = self.get_contours_outer(image)
            cnts, top_left, top_right, bottom_left, bottom_right = self.get_conf_outer_line(contours_)
            upper_center, lower_center, right_center, left_center = self.get_image_center(top_left, top_right, bottom_left, bottom_right)
            
            dis, rotation_status = self.get_rotate_direction(upper_center, lower_center)
            image = self.rotate_image(image, rotation_status)
            
            if dis < min_dis:
                min_dis = dis
                count = 0
            else:
                count += 1
              
            if dis < 15:
                break
            
            if dis < 20:
                break
            
            if dis < 25:
                break
            
            # Will automatically go out when there is no more update
            if count > 20 and dis < 35: 
                break  
            
            if count > 25:
                break
            
        print('Done rotating')    
        return image, central
    
    def check_distance_to_clasify_xy(self, z, val, idx=1): # 1 is for accessing y axis
        for key, x in z.items():  
            if abs(int(key) - np.squeeze(val)[idx]) <= 10:
                return key
        return None
    
    def get_horizontal_and_vertical_lines(self, cnts, idx=1, img_dim=[]):  
            
        # Provide a gap as for the starting point to start the line detection 
        gap = 20
        
        z = {} 
        for x in cnts:
            for y in x: 
                # Check the value to prevent from getting the border aligned with the image dimension
                if (np.squeeze(y)[0] > 0 + gap and np.squeeze(y)[0] < img_dim[1] - gap) and (np.squeeze(y)[1] > 0 + gap and np.squeeze(y)[1] < img_dim[0] - gap):
                    check = self.check_distance_to_clasify_xy(z, y, idx=idx)  # 1 is for accessing y axis
                    if check is None: 
                        # Use the y value as key to store all values within its range of lines
                        if np.squeeze(y)[idx] not in z:
                            z[np.squeeze(y)[idx]] = []
                        z[np.squeeze(y)[idx]].append(list(y))
                    else:
                        z[check].append(list(y))
         
        hor_lines = z.copy()
        for k, v in z.items(): 
            if len(v) < 100:  
                hor_lines.pop(k)
         
        return hor_lines     
    
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
                _lines_cp[key_2][len(np.squeeze(y)[:, idx])] = val
            else:
                _lines_cp[key_1][len(np.squeeze(y)[:, idx])] = val
          
        _line_1, _line_2 = [], []
        try:
            _line_1 = _lines_cp[key_1][max(list(_lines_cp[key_1].keys()))]
            _line_2 = _lines_cp[key_2][max(list(_lines_cp[key_2].keys()))]
        except:
            pass
        
        print(f'Get {status} detected ...')
        if status == 'VERTICAL_MEASUREMENT': 
            right_line, left_line = [], []
            left_line = _line_1
            right_line = _line_2
            return right_line, left_line
        else:
            top_line, bottom_line = [], []
            top_line = _line_1
            bottom_line = _line_2
            return top_line, bottom_line
    
    def get_outermost_line_to_measure(self, cnts, central, img_dim):
         
        # Get horizontal line of the card border contour
        hor_lines = self.get_horizontal_and_vertical_lines(cnts, idx=1, img_dim=img_dim) # 1 indicates y axis
        top_line, bottom_line = self.get_coordinate_to_measure(hor_lines, central, status='HORIZONTAL_MEASUREMENT')
        
        # Get vertical line of the card border contour
        ver_lines = self.get_horizontal_and_vertical_lines(cnts, idx=0, img_dim=img_dim) # 0 indicates x axis
        right_line, left_line = self.get_coordinate_to_measure(ver_lines, central, status='VERTICAL_MEASUREMENT')
          
        return top_line, bottom_line, right_line, left_line
    
    
    def image_segmentation(self, image, shadow=False):
        
        # Center image
        image, central = self.image_center(image)
        
        print('Extract the outer line of the card')
        # Get the outer line of the card
        # Get card border contour
        cnts, hierarchy = self.get_contour_full_border(image)
        top_line, bottom_line, right_line, left_line = self.get_outermost_line_to_measure(cnts, central, img_dim=image.shape)
        
        # Remove the center of the card to reduce some computation
        print('Remove the center of the card to reduce some computation')
        cut_inner_control = 1050
        gaps = 150
        new_a = image[int(central[1] - cut_inner_control):int(central[1] + cut_inner_control), int(central[0] - (cut_inner_control // 2 + gaps)):int(central[0] + (cut_inner_control // 2 + gaps))]
        
        # Rellocate the center of the rectangle that already cropped to white color
        print('Rellocate the center of the rectangle that already cropped to white color')
        image_1 = image.copy()  
        y_min, y_max, x_min, x_max = int(central[1] - cut_inner_control), int(central[1] + cut_inner_control), int(central[0] - (cut_inner_control // 2 + gaps)), int(central[0] + (cut_inner_control // 2 + gaps))
        rec_shape = new_a.shape
        image_1[y_min:y_max, x_min:x_max] = 255 * np.ones(rec_shape, np.uint8)
        
        # Remove some distance from the outer card to prevent of being re-detected
        print('Remove some distance from the outer card to prevent of being re-detected')
        cut_control = 20
        y_min_, y_max_, x_min_, x_max_ = int(top_line[1]) + cut_control, int(bottom_line[1]) - cut_control, int(left_line[0]) + cut_control, int(right_line[0]) - cut_control
        seg_img = image_1[y_min_:y_max_, x_min_:x_max_] # frame[y_min:y_max, x_min:x_max] 
          
        # Extract the contour of the inner card
        print('Extract the contour of the inner card')
        cnts = []
        if shadow:
            cnts = self.get_countour_shadow_card(seg_img)
        else:
            cnts = self.get_contours_all(seg_img) 
            
        # Draw the extracted inner line contours to an empty image frame 
        print('Draw the extracted inner line contours to an empty image frame')
        whiteFrame = 255 * np.ones(seg_img.shape, np.uint8)
        whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), 3)
        
        # Adjust the gap of the outer line to prevent from being re-detected
        print('Adjust the gap of the outer line to prevent from being re-detected')
        gap = 10
        whiteFrame = whiteFrame[gap:-gap, gap:-gap]
    
        # Adjust the extracted image into the original image dimension
        print('Adjust the extracted image into the original image dimension')
        whiteFrames = 255 * np.ones(image_1.shape, np.uint8)
        whiteFrames[y_min_ + gap:y_max_ - gap, x_min_ + gap:x_max_ - gap] = whiteFrame
          
        # Remove the inner rectangle of the card to prevent from misdetection of the inner lines
        print('Remove the inner rectangle of the card to prevent from misdetection of the inner lines')
        con = 50 
        whiteFrames[y_min - con:y_max  + con, x_min - con:x_max + con] = 255 * np.ones(whiteFrames[y_min - con:y_max  + con, x_min - con:x_max + con].shape, np.uint8)
          
        # Extract inner lines of the card
        print('Extract the inner lines of the card')
        gray = cv2.cvtColor(whiteFrames, cv2.COLOR_RGB2GRAY) 
        cnts, hierarchy = cv2.findContours(gray.copy(), 
                                           cv2.RETR_LIST ,
                                           cv2.CHAIN_APPROX_NONE)
        inter_top_line, inner_bottom_line, inner_right_line, inner_left_line = self.get_outermost_line_to_measure(cnts, central, img_dim=gray.shape)
     
        # To be used as outer
        outer_top_line, outer_bottom_line, outer_right_line, outer_left_line = [inter_top_line[0], top_line[1]], [inner_bottom_line[0], bottom_line[1]], [right_line[0], inner_right_line[1]], [left_line[0], inner_left_line[1]]
         
        return image, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line
    
    def plot_detection(self, image, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line, filename=''):
         
        whiteFrame = image.copy() # 255 * np.ones(image.shape, np.uint8) 
        text_color = (0, 0, 0)
        
        # Plot green lines on top
        whiteFrame = cv2.line(whiteFrame, (int(outer_top_line[0]), int(outer_top_line[1])), (int(inter_top_line[0]), int(inter_top_line[1])), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(outer_top_line[0]) - 50, int(outer_top_line[1])), (int(outer_top_line[0] + 50), int(outer_top_line[1])), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(inter_top_line[0]) - 50, int(inter_top_line[1])), (int(inter_top_line[0] + 50), int(inter_top_line[1])), (0, 253, 0), 5)
        pixels = math.sqrt(abs(int(outer_top_line[0]) - int(inter_top_line[0]))**2 + abs(int(outer_top_line[1]) - int(inter_top_line[1]))**2)
        mm = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(outer_top_line[0] + 60), int(outer_top_line[1])), cv2.FONT_HERSHEY_COMPLEX, 7, text_color, 3)
         
        # Plot green lines on bottom
        whiteFrame = cv2.line(whiteFrame, (int(outer_bottom_line[0]), int(outer_bottom_line[1])), (int(inner_bottom_line[0]), int(inner_bottom_line[1])), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(outer_bottom_line[0]) - 50, int(outer_bottom_line[1])), (int(outer_bottom_line[0] + 50), int(outer_bottom_line[1])), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(inner_bottom_line[0]) - 50, int(inner_bottom_line[1])), (int(inner_bottom_line[0] + 50), int(inner_bottom_line[1])), (0, 253, 0), 5)
        pixels = math.sqrt(abs(int(outer_bottom_line[0]) - int(inner_bottom_line[0]))**2 + abs(int(outer_bottom_line[1]) - int(inner_bottom_line[1]))**2)
        mm = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_bottom_line[0]), int(inner_bottom_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, 7, text_color, 3)
         
        # Plot green lines on right
        whiteFrame = cv2.line(whiteFrame, (int(outer_right_line[0]), int(outer_right_line[1])), (int(inner_right_line[0]), int(inner_right_line[1])), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(outer_right_line[0]), int(outer_right_line[1]) - 50), (int(outer_right_line[0]), int(outer_right_line[1] + 50)), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(inner_right_line[0]), int(inner_right_line[1] - 50)), (int(inner_right_line[0]), int(inner_right_line[1] + 50)), (0, 253, 0), 5)
        pixels = math.sqrt(abs(int(outer_right_line[0]) - int(inner_right_line[0]))**2 + abs(int(outer_right_line[1]) - int(inner_right_line[1]))**2)
        mm = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_right_line[0]), int(inner_right_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, 7, text_color, 3)
          
        # Plot green lines on left
        whiteFrame = cv2.line(whiteFrame, (int(outer_left_line[0]), int(outer_left_line[1])), (int(inner_left_line[0]), int(inner_left_line[1])), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(outer_left_line[0]), int(outer_left_line[1]) - 50), (int(outer_left_line[0]), int(outer_left_line[1] + 50)), (0, 253, 0), 5)
        whiteFrame = cv2.line(whiteFrame, (int(inner_left_line[0]), int(inner_left_line[1] - 50)), (int(inner_left_line[0]), int(inner_left_line[1] + 50)), (0, 253, 0), 5)
        pixels = math.sqrt(abs(int(outer_left_line[0]) - int(inner_left_line[0]))**2 + abs(int(outer_left_line[1]) - int(inner_left_line[1]))**2)
        mm = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_left_line[0]), int(inner_left_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, 7, text_color, 3)
           
        # Save image to directory
        cv2.imwrite(filename, whiteFrame)


if __name__ == '__main__':
    
    app = CardMeasurement()
    
    addr = 'D:\\Card-edge-measurement-release\\backside' 
    addr_to_save = 'D:\\Card-edge-measurement-release\\outputs'
    
    img_paths = os.listdir(addr) 
    for img_path in img_paths: 
        
        print(f'#### Processing image {img_path}')
        
        # Load the image,  
        image = cv2.imread(os.path.join(addr, img_path), 0)
        # image = cv2.imread(os.path.join(addr, "img-6-test.png"))
        
        shadow = False
        image, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line = app.image_segmentation(image, shadow=shadow)
        app.plot_detection(image, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line, filename=os.path.join(addr_to_save, img_path))
    
        print('Done\n\n')

 
