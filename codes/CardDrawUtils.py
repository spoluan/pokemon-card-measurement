# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:53:37 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import cv2
import math
from codes.RotationUtils import RotationUtils
from codes.LineUtils import LineUtils

class CardDrawUtils(object):
    
    def __init__(self):
        self.rotationUtils = RotationUtils()
        self.lineUtils = LineUtils()
        
    def plot_card_corner_detection(self, image, 
                                   outer_top_line, 
                                   outer_bottom_line, 
                                   outer_right_line, 
                                   outer_left_line, 
                                   inner_top_line, 
                                   inner_bottom_line, 
                                   inner_right_line, 
                                   inner_left_line):
        
        bold_size = 3
        whiteFrame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
        whiteFrame = cv2.cvtColor(whiteFrame, cv2.COLOR_GRAY2RGB) 
        
        top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner = [], [], [], []
        
        # Top left corner
        tltl, tltr, tlbl, tlbr = (int(outer_left_line[0]), int(outer_top_line[1])), (int(inner_left_line[0]), int(outer_top_line[1])), (int(outer_left_line[0]), int(inner_top_line[1])), (int(inner_left_line[0]), int(inner_top_line[1]))
        top_left_corner = [tltl, tltr, tlbl, tlbr ]
        whiteFrame = cv2.line(whiteFrame, tltl, tltr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, tltl, tlbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, tlbl, tlbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, tlbr, tltr, (0, 253, 0), bold_size) 
        
        # Top right corner
        trtl, trtr, trbl, trbr = (int(outer_right_line[0]), int(outer_top_line[1])), (int(inner_right_line[0]), int(outer_top_line[1])), (int(outer_right_line[0]), int(inner_top_line[1])), (int(inner_right_line[0]), int(inner_top_line[1]))
        top_right_corner = [trtl, trtr, trbl, trbr]
        whiteFrame = cv2.line(whiteFrame, trtl, trtr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, trtl, trbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, trbl, trbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, trbr, trtr, (0, 253, 0), bold_size)
        
        # Bottom left corner
        bltl, bltr, blbl, blbr = (int(outer_left_line[0]), int(inner_bottom_line[1])), (int(inner_left_line[0]), int(inner_bottom_line[1])), (int(outer_left_line[0]), int(outer_bottom_line[1])), (int(inner_left_line[0]), int(outer_bottom_line[1]))
        bottom_left_corner = [bltl, bltr, blbl, blbr]
        whiteFrame = cv2.line(whiteFrame, bltl, bltr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, bltr, blbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, blbr, blbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, blbl, bltl, (0, 253, 0), bold_size)
        
        # Bottom right corner
        brtl, brtr, brbl, brbr = (int(outer_right_line[0]), int(inner_bottom_line[1])), (int(inner_right_line[0]), int(inner_bottom_line[1])), (int(outer_right_line[0]), int(outer_bottom_line[1])), (int(inner_right_line[0]), int(outer_bottom_line[1]))
        bottom_right_corner = [brtl, brtr, brbl, brbr]
        whiteFrame = cv2.line(whiteFrame, brtl, brtr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, brtr, brbr, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, brbr, brbl, (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, brbl, brtl, (0, 253, 0), bold_size)
    
        return whiteFrame, [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]
    
    def plot_detection(self, image, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line, curvaturs, filename='', save_image=True):
         
        whiteFrame = image.copy() # 255 * np.ones(image.shape, np.uint8) 
        
        # Un-comment this for color output
        # is_color = False
        # if not is_color:
        #     whiteFrame = cv2.cvtColor(whiteFrame, cv2.COLOR_RGB2GRAY) 
        #     whiteFrame = cv2.cvtColor(whiteFrame, cv2.COLOR_GRAY2RGB) 
        
        text_color = (208, 67, 35)
        font_size = 2
        bold_size = 3
        
        top_dis, bottom_dis, right_dis, left_dis = 0, 0, 0, 0
        
        # Plot green lines on top
        whiteFrame = cv2.line(whiteFrame, (int(outer_top_line[0]), int(outer_top_line[1])), (int(inter_top_line[0]), int(inter_top_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_top_line[0]) - 50, int(outer_top_line[1])), (int(outer_top_line[0] + 50), int(outer_top_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inter_top_line[0]) - 50, int(inter_top_line[1])), (int(inter_top_line[0] + 50), int(inter_top_line[1])), (0, 253, 0), bold_size)
        # pixels = math.sqrt(abs(int(outer_top_line[0]) - int(inter_top_line[0]))**2 + abs(int(outer_top_line[1]) - int(inter_top_line[1]))**2)
        # mm = top_dis = round((pixels * 25.4) / 720, 2)
        mm = top_dis = self.lineUtils.get_mm(outer_top_line, inter_top_line)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(outer_top_line[0] + 60), int(outer_top_line[1])), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
         
        # Plot green lines on bottom
        whiteFrame = cv2.line(whiteFrame, (int(outer_bottom_line[0]), int(outer_bottom_line[1])), (int(inner_bottom_line[0]), int(inner_bottom_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_bottom_line[0]) - 50, int(outer_bottom_line[1])), (int(outer_bottom_line[0] + 50), int(outer_bottom_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inner_bottom_line[0]) - 50, int(inner_bottom_line[1])), (int(inner_bottom_line[0] + 50), int(inner_bottom_line[1])), (0, 253, 0), bold_size)
        # pixels = math.sqrt(abs(int(outer_bottom_line[0]) - int(inner_bottom_line[0]))**2 + abs(int(outer_bottom_line[1]) - int(inner_bottom_line[1]))**2)
        # mm = bottom_dis = round((pixels * 25.4) / 720, 2)
        mm = bottom_dis = self.lineUtils.get_mm(outer_bottom_line, inner_bottom_line)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_bottom_line[0]), int(inner_bottom_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
         
        # Plot green lines on right
        whiteFrame = cv2.line(whiteFrame, (int(outer_right_line[0]), int(outer_right_line[1])), (int(inner_right_line[0]), int(inner_right_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_right_line[0]), int(outer_right_line[1]) - 50), (int(outer_right_line[0]), int(outer_right_line[1] + 50)), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inner_right_line[0]), int(inner_right_line[1] - 50)), (int(inner_right_line[0]), int(inner_right_line[1] + 50)), (0, 253, 0), bold_size)
        # pixels = math.sqrt(abs(int(outer_right_line[0]) - int(inner_right_line[0]))**2 + abs(int(outer_right_line[1]) - int(inner_right_line[1]))**2)
        # mm = right_dis = round((pixels * 25.4) / 720, 2)
        mm = right_dis = self.lineUtils.get_mm(outer_right_line, inner_right_line)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_right_line[0]), int(inner_right_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
          
        # Plot green lines on left
        whiteFrame = cv2.line(whiteFrame, (int(outer_left_line[0]), int(outer_left_line[1])), (int(inner_left_line[0]), int(inner_left_line[1])), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(outer_left_line[0]), int(outer_left_line[1]) - 50), (int(outer_left_line[0]), int(outer_left_line[1] + 50)), (0, 253, 0), bold_size)
        whiteFrame = cv2.line(whiteFrame, (int(inner_left_line[0]), int(inner_left_line[1] - 50)), (int(inner_left_line[0]), int(inner_left_line[1] + 50)), (0, 253, 0), bold_size)
        # pixels = math.sqrt(abs(int(outer_left_line[0]) - int(inner_left_line[0]))**2 + abs(int(outer_left_line[1]) - int(inner_left_line[1]))**2)
        # mm = left_dis = round((pixels * 25.4) / 720, 2)
        mm = right_dis = self.lineUtils.get_mm(outer_left_line, inner_left_line)
        whiteFrame = cv2.putText(whiteFrame, f"{mm} mm", (int(inner_left_line[0]), int(inner_left_line[1]) - 60), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
           
        # Plot the curvaturs values
        curvature_top_left_corner, curvature_top_right_corner, curvature_bottom_left_corner, curvature_bottom_right_corner = curvaturs
        whiteFrame = cv2.putText(whiteFrame, f"{round(curvature_top_right_corner, 4)}", (int(outer_left_line[0]), int(outer_top_line[1])), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
        whiteFrame = cv2.putText(whiteFrame, f"{round(curvature_top_left_corner, 4)}", (int(outer_right_line[0]), int(outer_top_line[1])), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
        whiteFrame = cv2.putText(whiteFrame, f"{round(curvature_bottom_left_corner, 4)}", (int(outer_left_line[0]), int(outer_bottom_line[1]) + 50), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
        whiteFrame = cv2.putText(whiteFrame, f"{round(curvature_bottom_right_corner, 4)}", (int(outer_right_line[0]), int(outer_bottom_line[1]) + 50), cv2.FONT_HERSHEY_COMPLEX, font_size, text_color, 3)
         
        # If the image is potrait rotate it 
        whiteFrame, potrait_status = self.rotationUtils.is_potrait_then_rotate(whiteFrame, rotate_status='END')
          
        # Save image to directory
        if save_image: 
            
            cv2.imwrite(filename, whiteFrame)
        
        return top_dis, bottom_dis, right_dis, left_dis