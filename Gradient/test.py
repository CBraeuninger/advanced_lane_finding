'''
Created on 28 sept. 2019

@author: cbraeuninger
'''

import matplotlib.image as mpimg
import glob
import os
from Gradient import combine_gradient

#import images
#read in all images with names with pattern *.jpg
images = glob.glob('../test_images/*.jpg')

# loop over images and undistort them
for file_name in images:
    
    img = mpimg.imread(file_name)
    combined = combine_gradient(img)
    #save images
    if not os.path.exists("../output_images/gradient_images"):
        os.mkdir("../output_images/gradient_images")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/gradient_images", root + "-gradient" + ext)
    #save the result image
    mpimg.imsave(result_filename, combined, cmap='gray')