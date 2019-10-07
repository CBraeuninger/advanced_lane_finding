'''
Created on 1 oct. 2019

@author: cbraeuninger
'''

import matplotlib.pyplot as plt
from HLS_select import hls_select
import matplotlib.image as mpimg
import cv2
#import numpy as np
#import random

file_name = input("Give image path please:")
img = mpimg.imread(file_name)
hls = hls_select(img, (120,255), 'rgb')

# binary = np.zeros_like(hls, dtype='float32')
# for (i,j), val in np.ndenumerate(binary):
#     binary[i,j] = random.randint(0,1)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

cv2.namedWindow('test')
cv2.imshow('test', hls)
cv2.waitKey(0)
cv2.destroyWindow('test')
cv2.imwrite("../output_images/test/test.jpg", hls)
