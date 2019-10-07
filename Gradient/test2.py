'''
Created on 4 oct. 2019

@author: cbraeuninger
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from Gradient import combine_gradient

#file_name = input("Give image path please:")
img = mpimg.imread("../test_images/test2.jpg")
grad = combine_gradient(img)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
cv2.namedWindow('test')
cv2.imshow('test', grad)
cv2.waitKey(0)
cv2.destroyWindow('test')