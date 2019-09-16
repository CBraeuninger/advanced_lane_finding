'''
Created on 16 sept. 2019

@author: cbraeuninger
'''

import cv2
import glob
import numpy as np

def calibrate_camera():
    
    #Arrays to store the object points and image points for all calibration images
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane
    
    #The chess board has 9*6 inner points
    objp = np.zeros((6*8,3), np.float32)
    
    #mgrid[0:9,0:6] returns an array ([[[0,0,0,0,0,0], [1,1,1,1,1,1], ..., [8,8,8,8,8,8]],
    # [[0,1,2,3,4,5], ...[0,1,2,3,4,5]]])
    # Transpose returns an array ([[[0 0], [1 0], ..., [8,0]], [[0 1], [1 1], ...[8,1]], ...,
    # [[0 5], ..., [8 5]]])
    # Then reshape it to three columns and inferred number of rows
    objp = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    #read in all images with names with pattern calibration*.jpg
    cal_images = glob.glob('../camera_cal/calibration*.jpg')
    
    #loop over calibration images to compute camera matrix and distortion coefficients
    
    