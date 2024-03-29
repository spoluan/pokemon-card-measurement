# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:29:59 2022

@author: Sevendi Eldrige Rifki Poluan  
"""

import os
# os.chdir("D:\\post\\pokemon-card-measurement")  
import cv2    
from codes.FileUtils import FileUtils
from codes.RotationUtils import RotationUtils
from codes.ContourDetectionUtils import ContourDetectionUtils
from codes.LineUtils import LineUtils
from codes.CardRemovalUtils import CardRemovalUtils
from codes.CardDrawUtils import CardDrawUtils
from codes.CornerMeasurement import CornerMeasurement
from codes.ClientSocket import Client
import time
 
class CardMeasurement(object):
    
    def __init__(self):
        
        self.fileUtils = FileUtils()  
        self.rotationUtils = RotationUtils() 
        self.contourDetectionUtils = ContourDetectionUtils()
        self.lineUtils = LineUtils()
        self.cardRemovalUtils = CardRemovalUtils()
        self.cardDrawUtils = CardDrawUtils()
        self.cornerMeasurement = CornerMeasurement()
        self.client = Client()
    
    def image_segmentation_backside_pokemon_card(self, image, img_path): 
          
        # If the image is potrait rotate it 
        image, potrait_status = self.rotationUtils.is_potrait_then_rotate(image, rotate_status='START')
        
        # Rotate the image to the center 
        image, central = self.rotationUtils.image_auto_rotation(image)

        # preprocessing
        image = self.lineUtils.image_preprocessing(image, central, is_ouside=True, card_type=img_path)
        
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
      
    def process(self, addr, img_paths, addr_to_save, results_path):
        for count, img_path in enumerate(img_paths):  
            
            # Check if need to stop forcely
            if not self.fileUtils.is_to_stop(results_path, 'running_status.txt'):
                
                # Maintain the running status 
                print(f'#### Processing image {img_path} {count} from {len(img_paths)}') 
                self.fileUtils.update_information(results_path, 'current_process.txt', f'{img_path} > {count + 1} from {len(img_paths)}')
                
                time.sleep(2)

                # Load the image,  
                image = cv2.imread(os.path.join(addr, img_path), 1) # 0 | 1 = GRAY | RGB
                 
                try: 

                    # continue processing 
                    image, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line = self.image_segmentation_backside_pokemon_card(image, img_path)
                    
                    # Draw cornering corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]
                    image_with_corner, corners = self.cardDrawUtils.plot_card_corner_detection(image, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line)
                    
                    # Compute the cornering
                    curvatories = self.cornerMeasurement.extract_corner(image, corners) 
                    curvaturs = self.cornerMeasurement.get_curvatories(curvatories)
                    curvature_top_left_corner, curvature_top_right_corner, curvature_bottom_left_corner, curvature_bottom_right_corner = curvaturs
 
                    # Draw the measurement
                    top_dis, bottom_dis, right_dis, left_dis = self.cardDrawUtils.plot_detection(image_with_corner, potrait_status, outer_top_line, outer_bottom_line, outer_right_line, outer_left_line, inter_top_line, inner_bottom_line, inner_right_line, inner_left_line, curvaturs, filename=os.path.join(addr_to_save, img_path), save_image=True)
                    self.fileUtils.write_results([img_path, top_dis, bottom_dis, right_dis, left_dis, curvature_top_left_corner, curvature_top_right_corner, curvature_bottom_left_corner, curvature_bottom_right_corner], results_path, write_status='a+')
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
        addr = os.path.join(path, './sources')  
        addr_to_save = os.path.join(path, './outputs')
        results_path = os.path.join(path, './results')
        img_paths = [x for x in os.listdir(addr)[:] if '' in x][:]  

        print('IMAGE PATH TO DETECT', img_paths)

        # First run set default
        self.fileUtils.update_information(results_path, 'current_process.txt', '')  
        self.fileUtils.update_information(results_path, 'results.txt', '') 
        
        # Maintain the running status 
        self.fileUtils.update_information(results_path, 'running_status.txt', 'RUNNING')
        
        # Clear the results file for the first run
        self.fileUtils.write_results([], results_path, skipped=False)
        self.fileUtils.write_results([], results_path, skipped=True)
        self.fileUtils.prepare_output_dir(addr_to_save) # Remove the existing output image files first 
        
        # Call card processing 
        self.process(addr, img_paths, addr_to_save, results_path)
            
        # Maintain the running status 
        self.fileUtils.update_information(results_path, 'running_status.txt', 'STOP')
        self.fileUtils.update_information(results_path, 'current_process.txt', '')

        # STOP THE PROGRAM AND SERVER WHEN ALL IS FINISHED
        try:
            print('Connecting to the client ...')
            self.client.connect() 
            time.sleep(10) 
            print('Trying to force stop the server ...')
            self.client.send_stop()
            time.sleep(1) 
            self.client.force_stop()
            
        except Exception as t:
            print(t)
    
if __name__ == '__main__':
    app = CardMeasurement()
    app.main()