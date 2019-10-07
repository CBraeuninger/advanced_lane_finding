'''
Created on 29 sept. 2019

@author: cbraeuninger
'''
import cv2
from Gradient import combine_gradient

class GradientParameterFinder:
    def __init__(self, image, x_tr_min=0, x_tr_max=255, y_tr_min=0, y_tr_max=255, mag_tr_min=0, mag_tr_max=255, dir_tr_min=0, dir_tr_max=255):
        self.original = image
        self._x_tr_min = x_tr_min
        self._x_tr_max = x_tr_max
        self._y_tr_min = y_tr_min
        self._y_tr_max = y_tr_max
        self._mag_tr_min = mag_tr_min
        self._mag_tr_max = mag_tr_max
        self._dir_tr_min = dir_tr_min
        self._dir_tr_max = dir_tr_max        

        def onchange_x_tr_min(pos):
            self._x_tr_min = pos
            self._render()

        def onchange_x_tr_max(pos):
            self._x_tr_max = pos
            self._render()

        def onchange_y_tr_min(pos):
            self._y_tr_min = pos
            self._render()
           
        def onchange_y_tr_max(pos):
            self._y_tr_max = pos
            self._render() 
            
        def onchange_mag_tr_min(pos):
            self._mag_tr_min = pos
            self._render()
         
        def onchange_mag_tr_max(pos):
            self._mag_tr_max = pos
            self._render()
            
        def onchange_dir_tr_min(pos):
            self._dir_tr_min = pos
            self._render()   
            
        def onchange_dir_tr_max(pos):
            self._dir_tr_max = pos
            self._render() 

        cv2.namedWindow('edges')

        cv2.createTrackbar('min. x-grad.', 'edges', self._x_tr_min, 255, onchange_x_tr_min)
        cv2.createTrackbar('max. x-grad', 'edges', self._x_tr_max, 255, onchange_x_tr_max)
        cv2.createTrackbar('min. y-grad.', 'edges', self._y_tr_min, 255, onchange_y_tr_min)
        cv2.createTrackbar('max. y-grad.', 'edges', self._y_tr_max, 255, onchange_y_tr_max)
        cv2.createTrackbar('min. grad. mag.', 'edges', self._mag_tr_min, 255, onchange_mag_tr_min)
        cv2.createTrackbar('max. grad. mag.', 'edges', self._mag_tr_max, 255, onchange_mag_tr_max)
        cv2.createTrackbar('min. grad. dir.', 'edges', self._dir_tr_min, 255, onchange_dir_tr_min)
        cv2.createTrackbar('max. grad. dir.', 'edges', self._dir_tr_max, 255, onchange_dir_tr_max)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('original')

    def _render(self):
        
        self._edge_img = combine_gradient(self.original, 3, (self._x_tr_min,self._x_tr_max), (self._y_tr_min,self._y_tr_max), (self._mag_tr_min, self._mag_tr_max), (0.1*self._dir_tr_min,0.1*self._dir_tr_max))
        cv2.imshow('original', self.original)
        cv2.imshow('edges', self._edge_img)
        
    def x_tr_min(self):
        return self._x_tr_min
    
    def x_tr_max(self):
        return self._x_tr_max
    
    def y_tr_min(self):
        return self._y_tr_min
    
    def y_tr_max(self):
        return self._y_tr_max
    
    def mag_tr_min(self):
        return self._mag_tr_min
    
    def mag_tr_max(self):
        return self._mag_tr_max
    
    def dir_tr_min(self):
        return self._dir_tr_min
    
    def dir_tr_max(self):
        return self._dir_tr_max
        