# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:53:08 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import cv2
import numpy as np
from PIL import Image

class ContourDetectionUtils(object):
    
    def get_contours_outer(self, image):
        # Im read
        # import os 
        # addr = 'D:\\post\\pokemon-card-measurement\\Datasets\\data'
        # img_path = '4.jpg'  #, gx_10, gx_47, gx_8 
        # addr_to_save = './outputs'
        # image = cv2.imread(os.path.join(addr, img_path), 1)
        
        
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
        
        # whiteFrame = 255 * np.ones(image.shape, np.uint8)
        # whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), -1)
        # Image.fromarray(whiteFrame).convert('RGB').save(os.path.join(addr_to_save, f'{img_path.split(".")[0]}_out.jpg'))
   
        return cnts, hierarchy

    def get_contours_outer_vbeta(self, image):
        
        # Im read
        # import os 
        # addr = 'D:\\post\\pokemon-card-measurement\\Datasets\\data'
        # img_path = '4.jpg'  #, gx_10, gx_47, gx_8 
        # addr_to_save = './outputs'
        # image = cv2.imread(os.path.join(addr, img_path), 1)
        
        (lower, upper) = ([140, 140, 140], [255, 255, 255])
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper) 
        # output = cv2.bitwise_and(img, img, mask = mask)

        _, edged = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY_INV) # Best 175

        cnts, hierarchy = cv2.findContours(edged.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
         
        whiteFrame = 255 * np.ones(edged.shape, np.uint8)
        whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), -1)

        CC = cv2.Canny(np.array(whiteFrame), 0, 180)

        cnts, hierarchy = cv2.findContours(CC.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        whiteFrame = 255 * np.ones(image.shape, np.uint8) 
        # whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), -1)  
        # Image.fromarray(whiteFrame).convert('RGB').save(os.path.join(addr_to_save, f'{img_path.split(".")[0]}_out.jpg'))
   
        return cnts, hierarchy
    
    
    # Use for extracting the coordinate of the outermost
    def get_contour_full_border(self, image): 
        
        # import os 
        # addr = 'D:\\post\\pokemon-card-measurement\\Datasets\\data'
        # img_path = '4.jpg'  #, gx_10, gx_47, gx_8 
        # addr_to_save = './outputs'
        # image = cv2.imread(os.path.join(addr, img_path), 1)
         
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        _, edged = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # Update the value to adjust
        _, thresh = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 2))
        detect_horizontal = cv2.morphologyEx(close, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (0, 0, 0), -1)

        # Find vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 35))
        detect_vertical = cv2.morphologyEx(close, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (0, 0, 0), -1)
            
        CC = cv2.Canny(np.array(image), 0, 100)
          
        h, threshed = cv2.threshold(CC, 100, 255, cv2.THRESH_OTSU)  
        cnts, hierarchy = cv2.findContours(threshed.copy(), 
                                           cv2.RETR_LIST ,
                                           cv2.CHAIN_APPROX_NONE)  
        
        # whiteFrame = 255 * np.ones(threshed.shape, np.uint8)
        # whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), -1)  
        # cv2.imwrite(os.path.join(addr_to_save, img_path), whiteFrame)
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
        
        # whiteFrame = 255 * np.ones(seg_img.shape, np.uint8)
        # whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), 3)
        # plt.imshow(whiteFrame) 
        
        return cnts 
    
    # Use for extracting the inner border line (option one) - old version
    def get_contours_inner_vbeta(self, seg_img): 
        print('Extract the contour of the inner card')
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
        
        # Draw the extracted inner line contours to an empty image frame 
        print('Draw the extracted inner line contours to an empty image frame')
        whiteFrame = 255 * np.ones(seg_img.shape, np.uint8)
        whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), 3)
        
        # Apply the gap of the outer line to prevent from being re-detected
        print('Apply the gap of the outer line to prevent from being re-detected')
        gap_removal = 10
        whiteFrame = whiteFrame[gap_removal:-gap_removal, gap_removal:-gap_removal]
        
        return whiteFrame, cnts, gap_removal
    
    def get_contours_inner(self, seg_img):
        print('Extract the contour of the inner card')
        gray = self.sobel_algorithm(seg_img)  
        
        cnts, hierarchy = cv2.findContours(gray.copy(), 
                                           cv2.RETR_CCOMP ,
                                           cv2.CHAIN_APPROX_NONE) 
        
        
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True) 
        
        
        # Draw the extracted inner line contours to an empty image frame 
        print('Draw the extracted inner line contours to an empty image frame')
        whiteFrame = 255 * np.ones(seg_img.shape, np.uint8)
        whiteFrame = cv2.drawContours(whiteFrame, cnts, -1, (0, 0, 0), 1)
        
        # Apply the gap of the outer line to prevent from being re-detected
        print('Apply the gap of the outer line to prevent from being re-detected')
        gap_removal = 20
        whiteFrame = whiteFrame[gap_removal:-gap_removal, gap_removal:-gap_removal]
        
        return whiteFrame, cnts, gap_removal 
 
    
    def sobel_algorithm(self, matrix): 
        
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
        
        # C = cv2.Canny(np.array(mag), 0, 180)
        _, thresh = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY_INV)
        gray = cv2.cvtColor(np.array(thresh), cv2.COLOR_RGB2GRAY) 
        
        # rgb = cv2.cvtColor(whiteFrame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(addr_to_save, img_path), rgb)
        
        return gray