'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import glob
import os
import matplotlib.image as mpimg
import cv2
from PerspectiveTransform import doPerspectiveTransform


#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:
    
    img = cv2.imread(file_name, 0)#optional argument 0 is needed to read the image as a greyscale image
    warped = doPerspectiveTransform(img)
    #save images
    if not os.path.exists("../output_images/warped"):
        os.mkdir("../output_images/warped")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/warped", root + "-warped" + ext)
    #save the result image
    mpimg.imsave(result_filename, warped, cmap='gray')