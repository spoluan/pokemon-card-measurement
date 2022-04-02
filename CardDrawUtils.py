# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:53:37 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import cv2
import math
from RotationUtils import RotationUtils

class CardDrawUtils(object):
    
    def __init__(self):
        self.rotationUtils = RotationUtils()
        
    def plot_card_corner_detection(self, image, 
                                   outer_top_line, 
                                   outer_bottom_line, 
                                   outer_right_line, 
                                   outer_left_line, 
                                   inner_top_line, 
                                   inner_bottom_line, 
                                   inner_right_line, 
                                   inner_left_line):
        
        bold_size = 1
        
        # Top left corner
        tltl, tltr, tlbl, tlbr = (int(outer_left_line[0]), int(outer_top_line[1])), (int(inner_left_line[0], int(outer_top_line[1]))), (int(outer_left_line[0]), int(inner_top_line[1])), (int(inner_left_line[0]), int(inner_top_line[1]))
        whiteFrame = cv2.line(image, tltl, tltr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, tltl, tlbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, tlbl, tlbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, tlbr, tltr, (0, 253, 0), bold_size)
        
        # Top right corner
        trtl, trtr, trbl, trbr = (int(outer_right_line[0]), int(outer_top_line[1])), (int(inner_right_line[0], int(outer_top_line[1]))), (int(outer_right_line[0]), int(inner_top_line[1])), (int(inner_right_line[0]), int(inner_top_line[1]))
        whiteFrame = cv2.line(whiteFrame, trtl, trtr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, trtl, trbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, trbl, trbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, trbr, trtr, (0, 253, 0), bold_size)
        
        # Bottom left corner
        bltl, bltr, blbl, blbr = (int(outer_left_line[0]), int(outer_bottom_line[1])), (int(inner_left_line[0], int(outer_bottom_line[1]))), (int(outer_left_line[0]), int(inner_bottom_line[1])), (int(inner_left_line[0]), int(inner_bottom_line[1]))
        whiteFrame = cv2.line(whiteFrame, bltl, bltr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, bltr, blbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, blbl, blbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, blbr, bltr, (0, 253, 0), bold_size)
        
        # Bottom right corner
        brtl, brtr, brbl, brbr = (int(outer_left_line[0]), int(outer_bottom_line[1])), (int(inner_right_line[0], int(outer_bottom_line[1]))), (int(outer_right_line[0]), int(inner_bottom_line[1])), (int(inner_right_line[0]), int(inner_bottom_line[1]))
        whiteFrame = cv2.line(whiteFrame, brtl, brtr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, brtr, brbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, brbl, brbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, brbr, brtr, (0, 253, 0), bold_size)
    
        return whiteFrame
    
    def plot_detection(self, image, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line, filename='', save_image=True):
         
        whiteFrame = image.copy() # 255 * np.ones(image.shape, np.uint8) 
        
        # Un-comment this for color output
        is_color = False
        if not is_color:
            whiteFrame = cv2.cvtColor(whiteFrame, cv2.COLOR_RGB2GRAY) 
            whiteFrame = cv2.cvtColor(whiteFrame, cv2.COLOR_GRAY2RGB) 
        
        text_color = (208, 67, 35)
        font_size = 2
        bold_size = 3
        
        top_dis, bottom_dis, right_dis, left_dis = 0, 0, 0, 0
        
        # Plot green lines on top
        whiteFrame = cv2.line(whiteFrame, (int(outer_top_line[0]), int(outer_top_line[1])), (int(inter_top_line[0]), int(inter_top_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_top_line[0]) - 50, int(outer_top_line[1])), (int(outer_top_line[0] + 50), int(outer_top_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inter_top_line[0]) - 50, int(inter_top_line[1])), (int(inter_top_line[0] + 50), int(inter_top_line[1])), (0, 253, 0), bold_size)
        pixels = math.sqrt(abs(int(outer_top_line[0]) - int(inter_top_line[0]))**2 + abs(int(outer_top_line[1]) - int(inter_top_line[1]))**2)
        mm = top_dis = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(outer_top_line[0] + 60), int(outer_top_line[1])), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
         
        # Plot green lines on bottom
        whiteFrame = cv2.line(whiteFrame, (int(outer_bottom_line[0]), int(outer_bottom_line[1])), (int(inner_bottom_line[0]), int(inner_bottom_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_bottom_line[0]) - 50, int(outer_bottom_line[1])), (int(outer_bottom_line[0] + 50), int(outer_bottom_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inner_bottom_line[0]) - 50, int(inner_bottom_line[1])), (int(inner_bottom_line[0] + 50), int(inner_bottom_line[1])), (0, 253, 0), bold_size)
        pixels = math.sqrt(abs(int(outer_bottom_line[0]) - int(inner_bottom_line[0]))**2 + abs(int(outer_bottom_line[1]) - int(inner_bottom_line[1]))**2)
        mm = bottom_dis = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_bottom_line[0]), int(inner_bottom_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
         
        # Plot green lines on right
        whiteFrame = cv2.line(whiteFrame, (int(outer_right_line[0]), int(outer_right_line[1])), (int(inner_right_line[0]), int(inner_right_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_right_line[0]), int(outer_right_line[1]) - 50), (int(outer_right_line[0]), int(outer_right_line[1] + 50)), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inner_right_line[0]), int(inner_right_line[1] - 50)), (int(inner_right_line[0]), int(inner_right_line[1] + 50)), (0, 253, 0), bold_size)
        pixels = math.sqrt(abs(int(outer_right_line[0]) - int(inner_right_line[0]))**2 + abs(int(outer_right_line[1]) - int(inner_right_line[1]))**2)
        mm = right_dis = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_right_line[0]), int(inner_right_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
          
        # Plot green lines on left
        whiteFrame = cv2.line(whiteFrame, (int(outer_left_line[0]), int(outer_left_line[1])), (int(inner_left_line[0]), int(inner_left_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_left_line[0]), int(outer_left_line[1]) - 50), (int(outer_left_line[0]), int(outer_left_line[1] + 50)), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inner_left_line[0]), int(inner_left_line[1] - 50)), (int(inner_left_line[0]), int(inner_left_line[1] + 50)), (0, 253, 0), bold_size)
        pixels = math.sqrt(abs(int(outer_left_line[0]) - int(inner_left_line[0]))**2 + abs(int(outer_left_line[1]) - int(inner_left_line[1]))**2)
        mm = left_dis = round((pixels * 25.4) / 720, 2)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_left_line[0]), int(inner_left_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
           
        # If the image is potrait rotate it 
        whiteFrame, potrait_status = self.rotationUtils.is_potrait_then_rotate(whiteFrame, rotate_status='END')
          
        # Save image to directory
        if save_image: 
            
            cv2.imwrite(filename, whiteFrame)
        
        return top_dis, bottom_dis, right_dis, left_dis