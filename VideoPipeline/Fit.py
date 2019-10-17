'''
Created on 14 oct. 2019

@author: cbraeuninger
'''
import numpy as np

class Fit:

    def __init__(self):    
        self._l_fit = np.array([0,0,0])
        self._r_fit = np.array([0,0,0])
    
    def get_l_fit(self):
        return self._l_fit
    
    def get_r_fit(self):
        return self._r_fit
    
    def set_l_fit(self, new_l_fit):
        self._l_fit = new_l_fit
        
    def set_r_fit(self, new_r_fit):
        self._r_fit = new_r_fit
        
class FitReal:

    def __init__(self):    
        self._l_fit = np.array([0,0,0])
        self._r_fit = np.array([0,0,0])
    
    def get_l_fit(self):
        return self._l_fit
    
    def get_r_fit(self):
        return self._r_fit
    
    def set_l_fit(self, new_l_fit):
        self._l_fit = new_l_fit
        
    def set_r_fit(self, new_r_fit):
        self._r_fit = new_r_fit
        
class LineFit:
    
    #to initialize, choose the following two points on the left side of the image
    #and fit a line to them
    #x1 = 15%image_width, y1 = image_height
    #x2 = 45%image_width, y2 = 33%image_height
    #fit line to the followig points on the right side:
    #x1 = 85% image_width, y1 = image_height
    #x2 = 55% image_width, y2 = 33% image_height
    
    def __init__(self, img_width, img_height):

        self._left_line_fit = np.polyfit([img_height, 0.33*img_height], [0.15*img_width, 0.45*img_width], 1)
        self._right_line_fit = np.polyfit([img_height, 0.33*img_height], [0.85*img_height, 0.55*img_height],1)
        
    def get_left_line_fit(self):
        return self._left_line_fit
    
    def get_right_line_fit(self):
        return self._right_line_fit
    
    def set_left_line_fit(self, new_left_line_fit):
        self._left_line_fit = new_left_line_fit
        
    def set_right_line_fit(self, new_right_line_fit):
        self._right_line_fit = new_right_line_fit
        
    