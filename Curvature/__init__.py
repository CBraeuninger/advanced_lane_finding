'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
from LanePolyFit import findLanePixels, fitPolynomial

def calculateCurvature(fit, yeval, ym_per_pix):
          
    A = fit[0]
    B = fit[1]
    
    curvature = (1+(2*A*yeval*ym_per_pix+B)**2)**1.5/abs(2*A)
    
    return curvature
    

def realLaneCurvature(img, yeval):
    
    # Define conversions in x and y from pixels space to meters
    # Assuming the lane is about 30 meters long and 3.7 meters wide
    ym_per_pix = 30/img.shape[0] # meters per pixel in y dimension
    xm_per_pix = 3.7/img.shape[1] # meters per pixel in x dimension
    
    #find lane pixels
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, img = findLanePixels(img)
    
    #fit polynomial
    left_fit, right_fit, img = fitPolynomial(leftx*xm_per_pix, lefty*ym_per_pix, rightx*xm_per_pix, righty*ym_per_pix)
    
    #choose longer line segment and caculate curvature
    if l_right_seg > l_left_seg:
        curvature = calculateCurvature(right_fit, yeval, ym_per_pix)
    else:
        curvature = calculateCurvature(left_fit, yeval, ym_per_pix)
        
    return curvature