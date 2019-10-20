'''
Created on 20 oct. 2019

@author: cbraeuninger
'''
from Gradient import combine_gradient
from HLS_select import saturation_lightness_select, saturation_select
import cv2
import numpy as np

def createBinaryImage(img):
    
    '''
    output is an rgb image with only black and white pixels, not a "real" binary image
    '''
    #-------------------------------------- Combine gradient and lightness/saturation selection
    white = saturation_lightness_select(img, 150, 15, 'rgb')
    
    #-------------------------------------------------------- get gradient image
    grad = combine_gradient(img)
    
    #---------------------------------------------- combine with color selection
    color_grad = cv2.bitwise_and(grad, white)
    
    #------------- get saturation image and combine it with color/gradient image
    return cv2.bitwise_or(color_grad, saturation_select(img, (120,255), 120, 'rgb'))
