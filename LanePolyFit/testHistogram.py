'''
Created on 7 oct. 2019

@author: cbraeuninger
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

#read in image
img = cv2.imread("../output_images/final/straight_lines1-final.jpg")
#img = cv2.imread("../output_images/final/straight_lines2-final.jpg")

#take a histogram of the image
histogram = np.sum(img[:,:,0], axis=0)

fig, ax = plt.subplots()
x = range(histogram.shape[0])
ax.plot(x, histogram, color='green')
ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])
plt.show()

