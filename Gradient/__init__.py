'''
Created on 28 sept. 2019

@author: cbraeuninger
'''

import cv2
import numpy as np

def abs_sobel_threshold(img, sobel_kernel=3, orient='x', thresh=(0,255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (orient=='x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif (orient=='y'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled<=thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize =sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255*magnitude/np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled>=mag_thresh[0]) & (scaled<=mag_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir>=thresh[0]) & (grad_dir<=thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return binary_output

def combine_gradient(img, sobel_kernel=3, x_thresh=(20,100), y_thresh=(20,100), mag_thresh=(30,100), dir_thresh=(0.7,1.3)):
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(img, sobel_kernel, 'x', x_thresh)
    grady = abs_sobel_threshold(img, sobel_kernel, 'y', y_thresh)
    mag_binary = mag_threshold(img, sobel_kernel, mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel, dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((dir_binary == 1)&(mag_binary == 1))|((gradx==1) & (grady==1))] = 1
    
    return combined