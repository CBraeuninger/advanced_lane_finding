'''
Created on 6 oct. 2019

@author: cbraeuninger
'''

import glob
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
from Camera_Calibration import calibrate_camera
from Undistort_Image import undistort_image
from HLS_select import hls_select
from PerspectiveTransform import doPerspectiveTransform, warpImage
from VisualizationHelpers import saveResultImage
from LanePolyFit import findLanePixels, colorLanePixels
from FinalImage import finalImage

#-------------------------------------------------------- first calibrate camera
#get camera matrix and distortion coefficients
mtx, dist_coeff = calibrate_camera()

#----------------------------------------------------------------- import images

#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../test_images/*.jpg')

# loop over images
for file_name in images:
    
    img = mpimg.imread(file_name)
    
    #----------------------------------------------------------- undistort image
    undist = undistort_image(img, mtx, dist_coeff)    
    
    #--- convert to hls color space and apply threshold to generate binary image
    hls = hls_select(undist, (185,255), 120, 'rgb')
    
    #-------------------------------------- transform to bird's eye perspective
    warped, src, dst = doPerspectiveTransform(hls)
    
    #------------------------------------------------------------ fit polynomial
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, output_img = findLanePixels(warped)
    
    #-------------------------------- color detected lane pixels in warped image
    lanePix = colorLanePixels(np.zeros_like(warped), leftx, lefty, rightx, righty)
    
    #-------------------------------------------------------------- unwarp image
    lanePixUnwarped = warpImage(lanePix, dst, src)
    
    #superpose image of unwarped lane pixels image on original image and add curvature
    res_img = finalImage(img, lanePixUnwarped, img.shape[0], leftx, lefty, rightx, righty, l_left_seg, l_right_seg)
    
    #--------------------------------------------------------------- save images
    saveResultImage(res_img, "../output_images/final", file_name, "-final", True)
