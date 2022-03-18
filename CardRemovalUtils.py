# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:33:31 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import numpy as np

class CardRemovalUtils(object):
    
    def remove_card_center(self, image, central): 
        # Remove the center of the card to reduce some computation
        print('Remove the center of the card to reduce some computation')
        cut_inner_control = 1100
        gaps = 180
        new_a = image[int(central[1] - cut_inner_control):int(central[1] + cut_inner_control), int(central[0] - (cut_inner_control // 2 + gaps)):int(central[0] + (cut_inner_control // 2 + gaps))]
        
        # Rellocate the center of the rectangle that already cropped to white color
        print('Rellocate the center of the rectangle that already cropped to white color')
        image_1 = image.copy()  
        y_min, y_max, x_min, x_max = int(central[1] - cut_inner_control), int(central[1] + cut_inner_control), int(central[0] - (cut_inner_control // 2 + gaps)), int(central[0] + (cut_inner_control // 2 + gaps))
        rec_shape = new_a.shape
        image_1[y_min:y_max, x_min:x_max] = 255 * np.ones(rec_shape, np.uint8) 
        
        removed_b_box = y_min, y_max, x_min, x_max
        
        return image_1, removed_b_box
    
    def remove_some_slices_from_the_edge(self, image_1, top_line, bottom_line, right_line, left_line, cut_control=20):
        # Remove some distance from the outer card to prevent of being re-detected
        print('Remove some distance from the outer card to prevent of being re-detected')  
        seg_img, new_b_box = None, None
        try: 
            if type(top_line) == list:
                y_min_, y_max_, x_min_, x_max_ = int(top_line[1]) + cut_control, int(bottom_line[1]) - cut_control, int(left_line[0]) + cut_control, int(right_line[0]) - cut_control
               
            else:
                y_min_, y_max_, x_min_, x_max_ = top_line - cut_control, bottom_line + cut_control, right_line - cut_control, left_line + cut_control
        
            seg_img = image_1[y_min_:y_max_, x_min_:x_max_] # frame[y_min:y_max, x_min:x_max]  
            
            new_b_box = y_min_, y_max_, x_min_, x_max_
        except Exception as err:
            print('ISUES', err)
        return seg_img, new_b_box

    def frame_adjustment_to_the_original_size(self, image_size, whiteFrameWithCountour_src, removed_b_box, new_removed_b_box, gap_removal):
        y_min_, y_max_, x_min_, x_max_ = new_removed_b_box[0], new_removed_b_box[1], new_removed_b_box[2], new_removed_b_box[3] 
        print('Adjust the extracted image into the original image dimension')
        whiteFrames = 255 * np.ones(image_size, np.uint8)
        whiteFrames[y_min_ + gap_removal:y_max_ - gap_removal, x_min_ + gap_removal:x_max_ - gap_removal] = whiteFrameWithCountour_src
        
        # Remove the outer rectangle of the card to prevent from misdetection of the inner lines
        print('Remove the outer rectangle of the card to prevent from misdetection of the inner lines')
        con = 30 
        y_min, y_max, x_min, x_max = removed_b_box[0], removed_b_box[1], removed_b_box[2], removed_b_box[3] 
        whiteFrames[y_min - con:y_max  + con, x_min - con:x_max + con] = 255 * np.ones(whiteFrames[y_min - con:y_max  + con, x_min - con:x_max + con].shape, np.uint8)
          
        return whiteFrames