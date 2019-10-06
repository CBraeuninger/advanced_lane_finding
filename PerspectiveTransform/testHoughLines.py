'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import cv2
import glob
import os
import matplotlib.image as mpimg
from PerspectiveTransform import houghLinesDetection


#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    #read in example image
    img = cv2.imread(file_name, 0)
    
    #detect lines using Hough algorithm
    lines = houghLinesDetection(img)
    
    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #loop over all the lines and draw them on the image
    for line in lines:
        line_img = cv2.line(line_img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255,0,0), 5)
        
    #save images
    if not os.path.exists("../output_images/HoughLines"):
        os.mkdir("../output_images/HoughLines")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/HoughLines", root + "-hough" + ext)
    #save the result image
    mpimg.imsave(result_filename, line_img, cmap='gray')