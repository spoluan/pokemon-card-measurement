# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:29:59 2022

@author: Sevendi Eldrige Rifki Poluan  
"""

import os
# os.chdir("D:\\post\\pokemon-card-measurement")  
import cv2    
from FileUtils import FileUtils
from RotationUtils import RotationUtils
from ContourDetectionUtils import ContourDetectionUtils
from LineUtils import LineUtils
from CardRemovalUtils import CardRemovalUtils
from CardDrawUtils import CardDrawUtils
 
class CardMeasurement(object):
    
    def __init__(self):
        
        self.fileUtils = FileUtils()  
        self.rotationUtils = RotationUtils() 
        self.contourDetectionUtils = ContourDetectionUtils()
        self.lineUtils = LineUtils()
        self.cardRemovalUtils = CardRemovalUtils()
        self.cardDrawUtils = CardDrawUtils()
    
    def image_segmentation_backside_pokemon_card(self, image, img_path): 
          
        # If the image is potrait rotate it 
        image, potrait_status = self.rotationUtils.is_potrait_then_rotate(image, rotate_status='START')
        
        # Rotate the image to the center 
        image, central = self.rotationUtils.image_auto_rotation(image)
        
        img = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
        
        # Get the outer line of the card 
        cnts, top, bottom, right, left, \
            one_top, two_bottom, one_right, two_left, \
            top_line, bottom_line, right_line, left_line, central = self.lineUtils.get_outermost_line_to_measure(image, central, is_ouside=True, card_type=img_path)
         
        # Remove the center of the card to reduce some computation
        image_1, removed_b_box = self.cardRemovalUtils.remove_card_center(image, central)
        
        # Remove some distance from the outer card to prevent of being re-detected
        seg_img, new_removed_b_box = self.cardRemovalUtils.remove_some_slices_from_the_edge(image_1, top_line, bottom_line, right_line, left_line, cut_control=30)
        
        # Extract the contour of the inner card 
        whiteFrameWithCountour, cnts, gap_removal = self.contourDetectionUtils.get_contours_inner(seg_img) 
               
        # Adjust the extracted image into the original image dimension
        whiteFrames = self.cardRemovalUtils.frame_adjustment_to_the_original_size(image_1.shape, whiteFrameWithCountour, removed_b_box, new_removed_b_box, gap_removal)
          
        # Extract inner lines of the card 
        cnts, top, bottom, right, left, \
            one_top, two_bottom, one_right, two_left, \
            top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate, _ = self.lineUtils.get_outermost_line_to_measure(whiteFrames, central, is_binary=True, is_ouside=False, card_type=img_path)
       
        # To be used as outer and inner coordinates
        outer_top_line, outer_bottom_line, outer_right_line, outer_left_line = \
            [top_center_coordinate[0], top_line[1]], \
            [bottom_center_coordinate[0], bottom_line[1]], \
            [right_line[0], right_center_coordinate[1]], \
            [left_line[0], left_center_coordinate[1]]
        inner_top_line, inner_bottom_line, inner_right_line, inner_left_line = \
            top_center_coordinate, bottom_center_coordinate, right_center_coordinate, left_center_coordinate
        
        return img, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inner_top_line, inner_bottom_line, inner_right_line, inner_left_line
      
    def process(self, addr, img_paths, addr_to_save, results_path, card_model='backside_pokemon_card'):
        for count, img_path in enumerate(img_paths):  
            
            # Check if need to stop forcely
            if not self.fileUtils.is_to_stop(results_path, 'running_status.txt'):
                
                # Maintain the running status 
                print(f'#### Processing image {img_path} {count} from {len(img_paths)}') 
                self.fileUtils.update_information(results_path, 'current_process.txt', f'{img_path} > {count} from {len(img_paths)}')
                 
                # Load the image,  
                image = cv2.imread(os.path.join(addr, img_path), 1) # 0 | 1 = GRAY | RGB
                 
                try:
                    
                    # Different for every card model
                    if card_model == 'backside_pokemon_card':
                        image, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line = self.image_segmentation_backside_pokemon_card(image, img_path)
                        
                    top_dis, bottom_dis, right_dis, left_dis = self.cardDrawUtils.plot_detection(image, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line, filename=os.path.join(addr_to_save, img_path), save_image=True)
                    self.fileUtils.write_results([img_path, top_dis, bottom_dis, right_dis, left_dis], results_path, write_status='a+')
                except Exception as a:
                    print(a)
                    self.fileUtils.write_results([img_path], results_path, skipped=True, write_status='a+')
                    print(f'Imature detection of some border: image {img_path} is being skipped')
                
                print('Done\n\n')
            
            else:
                print('Force closed')
                break 
    
    def main(self):
        
        # Specify the path to save
        path = os.path.dirname(__file__)
        addr = os.path.join(path, './Datasets/data-fixed-detected') # ./sources' 
        addr_to_save = os.path.join(path, './outputs')
        results_path = os.path.join(path, './results')
        img_paths = [x for x in os.listdir(addr)[:] if '' in x] 
        
        # Maintain the running status 
        self.fileUtils.update_information(results_path, 'running_status.txt', 'RUNNING')
        
        # Clear the results file for the first run
        self.fileUtils.write_results([], results_path, skipped=False)
        self.fileUtils.write_results([], results_path, skipped=True)
        self.fileUtils.prepare_output_dir(addr_to_save) # Remove the existing output image files first 
        
        # Call card processing: backside_pokemon_card | ...
        self.process(addr, img_paths, addr_to_save, results_path, card_model='backside_pokemon_card')
            
        # Maintain the running status 
        self.fileUtils.update_information(results_path, 'running_status.txt', 'STOP')
        self.fileUtils.update_information(results_path, 'current_process.txt', '')
    
if __name__ == '__main__':
    app = CardMeasurement()
    app.main()
    