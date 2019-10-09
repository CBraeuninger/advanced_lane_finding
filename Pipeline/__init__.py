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
    hls = hls_select(undist, (185,255), 'rgb')
    
    #-------------------------------------- transform to bird's eye perspective
    warped, src, dst = doPerspectiveTransform(hls)
    
    #------------------------------------------------------------ fit polynomial
    retLanePix = findLanePixels(warped)
    
    #-------------------------------- color detected lane pixels in warped image
    lanePixMarked = colorLanePixels(np.zeros_like(warped), retLanePix[0], retLanePix[1], retLanePix[2], retLanePix[3])
    
    #-------------------------------------------------------------- unwarp image
    res_img = warpImage(lanePixMarked, dst, src)
    
    #--------------------------------------------------------------- save images
    saveResultImage(lanePixMarked, "../output_images/final", file_name, "-final", True)
