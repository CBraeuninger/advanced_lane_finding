'''
Created on 1 oct. 2019

@author: cbraeuninger
'''

import cv2
import numpy as np

# Function that thresholds the S-channel of an HLS image
def hls_select(img, s_thresh=(0, 255), l_thresh=255, color_space='bgr'):
    # Convert to HLS color space
    if color_space == 'bgr':
            grad = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif color_space == 'rgb':
        grad = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Select S channel
    S = grad[:,:,2]
    # Apply a threshold to the S channel
    binary_output = np.zeros_like(S, dtype='float32')
    binary_output[(S>s_thresh[0]) & (S<=s_thresh[1])]=1
    #Select also L (lightness) chanel
    L = grad[:,:,1]
    #Apply threshold on that too (remove pixels that are too dark)
    binary_output[(L<l_thresh)]=0
    # Transform binary image to RGB
    rgb = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    rgb[:,:,0] = binary_output*255
    rgb[:,:,1] = binary_output*255
    rgb[:,:,2] = binary_output*255
    
    return rgb