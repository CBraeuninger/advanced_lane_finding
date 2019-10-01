'''
Created on 1 oct. 2019

@author: cbraeuninger
'''
import glob
import os
import matplotlib.image as mpimg
from HLS_select import hls_select


#import images
#read in all images with names with pattern *.jpg
images = glob.glob('../test_images/*.jpg')

# loop over images and undistort them
for file_name in images:
    
    img = mpimg.imread(file_name)
    hls = hls_select(img, (120,255))
    #save images
    if not os.path.exists("../output_images/hls"):
        os.mkdir("../output_images/hls")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/hls", root + "-hls" + ext)
    #save the result image
    mpimg.imsave(result_filename, hls, cmap='gray')