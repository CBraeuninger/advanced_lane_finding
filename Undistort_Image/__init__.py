'''
Created on 25 sept. 2019

@author: cbraeuninger
'''
import cv2

'''
Undistorts image given a camera matrix and distortion coefficients
'''
def undistort_image(img, mtx, dist):
    
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist_img