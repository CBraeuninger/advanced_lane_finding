'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
from LanePolyFit import fitPolynomial
from VideoPipeline import config

def getConversion(img_height, leftx_base, rightx_base):
    # conversion factors for pixel/meters
    # y-direction:
    # the lane lines are about 30 m long
    # in the perspective transform we take about half of that and project it to the warped image
    ym_per_pix = 15/img_height
    
    # the lane is about 3.7 m wide
    # in the warped image that corresponds to the number of pixels between the left and right lane
    xm_per_pix = 3.7/(rightx_base - leftx_base)
    
    return ym_per_pix, xm_per_pix

def calculateCurvature(fit, yeval, ym_per_pix):
          
    A = fit[0]
    B = fit[1]
    
    curv = (1+(2*A*yeval*ym_per_pix+B)**2)**1.5/abs(2*A)
    
    return curv
    

def realLaneCurvature(yeval, leftx, lefty, rightx, righty, l_left_seg, l_right_seg, ym_per_pix, xm_per_pix):
    
    #fit polynomial
    left_fit, right_fit, img = fitPolynomial(leftx*xm_per_pix, lefty*ym_per_pix, rightx*xm_per_pix, righty*ym_per_pix, config.fitReal)
    
    #choose longer line segment and caculate curvature
    if l_right_seg > l_left_seg:
        curv = calculateCurvature(right_fit, yeval, ym_per_pix)
    else:
        curv = calculateCurvature(left_fit, yeval, ym_per_pix)
     
    config.fitReal.set_l_fit(left_fit)
    config.fitReal.set_r_fit(right_fit) 
        
    return curv

def distanceFromLane(img, src):
    
    #conversion factor for unwarped image
    #
    xm_per_pix = 3.7/(src[3][0]-src[0][0])
    
    #center of vehicle is at center of image
    x_veh = img.shape[1]/2*xm_per_pix
    
    #calculate middle of lane line
    x_middle = (0.5*src[0][0] + 0.5*src[3][0])*xm_per_pix
    
    #calculate offset
    offset = x_middle - x_veh
    
    return offset
    