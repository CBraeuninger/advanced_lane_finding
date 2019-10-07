'''
Created on 1 oct. 2019

@author: cbraeuninger
'''

import cv2
import numpy as np

# Function that thresholds the S-channel of an HLS image
def hls_select(img, thresh=(0, 255), color_space='bgr'):
    # 1) Convert to HLS color space
    if color_space == 'bgr':
            grad = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif color_space == 'rgb':
        grad = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Select S channel
    S = grad[:,:,2]
    # 3) Apply a threshold to the S channel
    binary_output = np.zeros_like(S, dtype='float32')
    binary_output[(S>thresh[0]) & (S<=thresh[1])]=1
    # 4) Transform binary image to RGB
    rgb = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    rgb[:,:,0] = binary_output*255
    rgb[:,:,1] = binary_output*255
    rgb[:,:,2] = binary_output*255
    
    return rgb