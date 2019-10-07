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
from PerspectiveTransform import doPerspectiveTransform

#-------------------------------------------------------- first calibrate camera
#get camera matrix and distortion coefficients
mtx, dst = calibrate_camera()

#----------------------------------------------------------------- import images

#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../test_images/*.jpg')

# loop over images
for file_name in images:
    
    img = cv2.imread(file_name)
    
    #----------------------------------------------------------- undistort image
    undist = undistort_image(img, mtx, dst)    
    
    #--- convert to hls color space and apply threshold to generate binary image
    hls = hls_select(undist, (185,255))
    
    #-------------------------------------- transform to bird's eye perspective
    warped = doPerspectiveTransform(hls)    
    
    #--------------------------------------------------------------- save images
    if not os.path.exists("../output_images/final"):
        os.mkdir("../output_images/final")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/final", root + "-final" + ext)
    #save the result image
    mpimg.imsave(result_filename, warped, cmap='gray')