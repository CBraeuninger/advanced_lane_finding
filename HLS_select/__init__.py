'''
Created on 1 oct. 2019

@author: cbraeuninger
'''

import cv2
import numpy as np

# Function that thresholds the S-channel of an HLS image
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Select S channel
    S = hls[:,:,2]
    # 3) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S>thresh[0]) & (S<=thresh[1])]=1
    # 4) Return a binary image of threshold result
    return binary_output