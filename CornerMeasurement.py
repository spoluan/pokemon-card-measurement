# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:20:54 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import cv2 
import os
import numpy as np
from Curvature import ComputeCurvature

class CornerMeasurement(object):
    
    def __init__(self):
        self.computeCurvature_top_left_corner = ComputeCurvature()
        self.computeCurvature_top_right_corner = ComputeCurvature()
        self.computeCurvature_bottom_left_corner = ComputeCurvature()
        self.computeCurvature_bottom_right_corner = ComputeCurvature()
    
    # corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]
    def extract_corner(self, image, corners):
         
        top_left_corner = corners[0]
        top_right_corner = corners[1]
        bottom_left_corner = corners[2]
        bottom_right_corner = corners[3]
         
        filter_top_left_corner = image[top_left_corner[1][1]:top_left_corner[2][1], top_left_corner[0][0]:top_left_corner[1][0]]
        filter_top_right_corner = image[top_right_corner[1][1]:top_right_corner[2][1], top_right_corner[1][0]:top_right_corner[0][0]]
        filter_bottom_left_corner = image[bottom_left_corner[1][1]:bottom_left_corner[2][1], bottom_left_corner[0][0]:bottom_left_corner[1][0]]
        filter_bottom_right_corner = image[bottom_right_corner[1][1]:bottom_right_corner[2][1], bottom_right_corner[1][0]:bottom_right_corner[0][0]]
          
        # EDGE FILTERING
        edge_filter_top_left_corner = self.edge_extraction(filter_top_left_corner)
        edge_filter_top_right_corner = self.edge_extraction(filter_top_right_corner)
        edge_filter_bottom_left_corner = self.edge_extraction(filter_bottom_left_corner) 
        edge_filter_bottom_right_corner = self.edge_extraction(filter_bottom_right_corner) 
         
        # SELECT ONLY THE CURVE LINE
        curve_filter_top_left_corner = self.get_curve_line(edge_filter_top_left_corner, top=True)
        curve_filter_top_right_corner = self.get_curve_line(edge_filter_top_right_corner, top=True)
        curve_filter_bottom_left_corner = self.get_curve_line(edge_filter_bottom_left_corner, top=False)
        curve_filter_bottom_right_corner = self.get_curve_line(edge_filter_bottom_right_corner, top=False)
         
        
        # PLOT THE RESULTS
        img_curve_curve_top_left_corner = self.plot_filtered_line(edge_filter_top_left_corner, curve_filter_top_left_corner)
        img_curve_filter_top_right_corner = self.plot_filtered_line(edge_filter_top_right_corner, curve_filter_top_right_corner)
        img_curve_filter_bottom_left_corner = self.plot_filtered_line(edge_filter_bottom_left_corner, curve_filter_bottom_left_corner)
        img_curve_filter_bottom_right_corner= self.plot_filtered_line(edge_filter_bottom_right_corner, curve_filter_bottom_right_corner)
        
        # PLOT ALL THE RESULS
        # data = [
        #           filter_top_left_corner,
        #           filter_top_right_corner,
        #           filter_bottom_left_corner,
        #           filter_bottom_right_corner,
        #               edge_filter_top_left_corner,
        #               edge_filter_top_right_corner,
        #               edge_filter_bottom_left_corner,
        #               edge_filter_bottom_right_corner,
        #                   img_curve_curve_top_left_corner,
        #                   img_curve_filter_top_right_corner,
        #                   img_curve_filter_bottom_left_corner,
        #                   img_curve_filter_bottom_right_corner
        #     ]
        # self.plots(*data)

        return [curve_filter_top_left_corner, curve_filter_top_right_corner, 
                    curve_filter_bottom_left_corner, curve_filter_bottom_right_corner]
        
    def plots(self, 
                  filter_top_left_corner,
                  filter_top_right_corner,
                  filter_bottom_left_corner,
                  filter_bottom_right_corner,
                      edge_filter_top_left_corner,
                      edge_filter_top_right_corner,
                      edge_filter_bottom_left_corner,
                      edge_filter_bottom_right_corner,
                          img_curve_curve_top_left_corner,
                          img_curve_filter_top_right_corner,
                          img_curve_filter_bottom_left_corner,
                          img_curve_filter_bottom_right_corner
              ):
        self.write_img(filter_top_left_corner, 'filter_top_left_corner', '1')
        self.write_img(filter_top_right_corner, 'filter_top_right_corner', '2')
        self.write_img(filter_bottom_left_corner, 'filter_bottom_left_corner', '3')
        self.write_img(filter_bottom_right_corner, 'filter_bottom_right_corner', '4')
        
        self.write_img(edge_filter_top_left_corner, 'edge_filter_top_left_corner', '11')
        self.write_img(edge_filter_top_right_corner, 'edge_filter_top_right_corner', '22')
        self.write_img(edge_filter_bottom_left_corner, 'edge_filter_bottom_left_corner', '33')
        self.write_img(edge_filter_bottom_right_corner, 'edge_filter_bottom_right_corner', '44')
        
        self.write_img(img_curve_curve_top_left_corner, 'img_curve_curve_top_left_corner', '111')
        self.write_img(img_curve_filter_top_right_corner, 'img_curve_filter_top_right_corner', '222')
        self.write_img(img_curve_filter_bottom_left_corner, 'img_curve_filter_bottom_left_corner', '333')
        self.write_img(img_curve_filter_bottom_right_corner, 'img_curve_filter_bottom_right_corner', '444')
         
    def plot_filtered_line(self, corner, hold):
        p = 0 * np.ones(corner.shape, np.uint8)
        for x in hold:
            try:
                p[x[0], x[1]] = 255
            except Exception as a:
                pass 
        return p
        
    def write_img(self, img, name, initial):
        addr_to_save = './outputs' 
        cv2.imwrite(os.path.join(addr_to_save, f'{initial}_results_{name}.jpg'), img)
        
    def edge_extraction(self, corner):
        return cv2.Canny(corner, 50, 100, None, 3)
     
    def get_curve_line(self, extracted_canny, top=True):   
        xx = [] 
        if top: 
            for x in range(extracted_canny.shape[1]):
                for id, y in enumerate(extracted_canny[:, x]):
                    if y > 200:
                        xx.append([x, id])   
                        break   
        else: 
            for x in range(extracted_canny.shape[1]):
                for id, y in enumerate(extracted_canny[:, x][::-1]):
                    if y > 200:
                        xx.append([x, extracted_canny.shape[0] - id])   
                        break    
        return xx
 
    # corner_coordinate_sets = [curve_filter_top_left_corner, curve_filter_top_right_corner, curve_filter_bottom_left_corner, curve_filter_bottom_right_corner]
    def get_curvatories(self, corner_coordinate_sets):

        curve_filter_top_left_corner = corner_coordinate_sets[0]
        curve_filter_top_right_corner = corner_coordinate_sets[1]
        curve_filter_bottom_left_corner = corner_coordinate_sets[2]
        curve_filter_bottom_right_corner = corner_coordinate_sets[3] 
          
        # GET CURVATORY VALUES
        curvature_top_left_corner = self.computeCurvature_top_left_corner.fit(np.array(curve_filter_top_left_corner)[:, 0], np.array(curve_filter_top_left_corner)[:, 1])
        curvature_top_right_corner = self.computeCurvature_top_right_corner.fit(np.array(curve_filter_top_right_corner)[:, 0], np.array(curve_filter_top_right_corner)[:, 1])
        curvature_bottom_left_corner = self.computeCurvature_bottom_left_corner.fit(np.array(curve_filter_bottom_left_corner)[:, 0], np.array(curve_filter_bottom_left_corner)[:, 1])
        curvature_bottom_right_corner = self.computeCurvature_bottom_right_corner.fit(np.array(curve_filter_bottom_right_corner)[:, 0], np.array(curve_filter_bottom_right_corner)[:, 1])
        
        # GET CURVE PLOTTING VALUES
        # x_y_curvature_top_left_corner = self.computeCurvature_top_left_corner.get_curve_plotting_values(self.computeCurvature_top_left_corner)
        # x_y_curvature_top_right_corner = self.computeCurvature_top_right_corner.get_curve_plotting_values(self.computeCurvature_top_right_corner)
        # x_y_curvature_bottom_left_corner = self.computeCurvature_bottom_left_corner.get_curve_plotting_values(self.computeCurvature_bottom_left_corner)
        # x_y_curvature_bottom_right_corner = self.computeCurvature_bottom_right_corner.get_curve_plotting_values(self.computeCurvature_bottom_right_corner)
          
         
        # SAVE PLOTING
        # self.computeCurvature_top_left_corner.curve_plot(curvature_top_left_corner, 
        #                                                  np.array(curve_filter_top_left_corner)[:, 0], 
        #                                                  np.array(curve_filter_top_left_corner)[:, 1], 
        #                                                  x_y_curvature_top_left_corner[0], 
        #                                                  x_y_curvature_top_left_corner[1], 
        #                                                  'filter_top_left_corner', 
        #                                                  '1111')
        # self.computeCurvature_top_right_corner.curve_plot(curvature_top_right_corner, 
        #                                                   np.array(curve_filter_top_right_corner)[:, 0], 
        #                                                   np.array(curve_filter_top_right_corner)[:, 1], 
        #                                                   x_y_curvature_top_right_corner[0], 
        #                                                   x_y_curvature_top_right_corner[1], 
        #                                                   'filter_top_right_corner', 
        #                                                   '2222')
        # self.computeCurvature_bottom_left_corner.curve_plot(curvature_bottom_left_corner, 
        #                                                   np.array(curve_filter_bottom_left_corner)[:, 0], 
        #                                                   np.array(curve_filter_bottom_left_corner)[:, 1], 
        #                                                   x_y_curvature_bottom_left_corner[0], 
        #                                                   x_y_curvature_bottom_left_corner[1], 
        #                                                   'filter_bottom_left_corner', 
        #                                                   '3333')
        # self.computeCurvature_bottom_right_corner.curve_plot(curvature_bottom_right_corner, 
        #                                                   np.array(curve_filter_bottom_right_corner)[:, 0], 
        #                                                   np.array(curve_filter_bottom_right_corner)[:, 1], 
        #                                                   x_y_curvature_bottom_right_corner[0], 
        #                                                   x_y_curvature_bottom_right_corner[1], 
        #                                                   'filter_bottom_right_corner', 
        #                                                   '4444')

        return [curvature_top_left_corner, curvature_top_right_corner,
                curvature_bottom_left_corner, curvature_bottom_right_corner]
        

    





