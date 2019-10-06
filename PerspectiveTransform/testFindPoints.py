'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
import os
import matplotlib.image as mpimg
from PerspectiveTransform import houghLinesDetection, findPoints

#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    #read in example image
    img = cv2.imread(file_name, 0)
    
    #do Hough transform
    lines = houghLinesDetection(img)
    #get source and destination points
    src, dst = findPoints(lines, img.shape[1], img.shape[0])
    
    #Transform image to RGB
    dots_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    #draw dot (small circle on image)
    #source points
    dots_img = cv2.circle(dots_img, (src[0][0], src[0][1]), 5, (255,0,0), -1)
    dots_img = cv2.circle(dots_img, (src[1][0], src[1][1]), 5, (255,0,0), -1)
    dots_img = cv2.circle(dots_img, (src[2][0], src[2][1]), 5, (255,0,0), -1)
    dots_img = cv2.circle(dots_img, (src[3][0], src[3][1]), 5, (255,0,0), -1)
    #destination points
    dots_img = cv2.circle(dots_img, (dst[0][0], dst[0][1]), 5, (0,255,0), -1)
    dots_img = cv2.circle(dots_img, (dst[1][0], dst[1][1]), 5, (0,255,0), -1)
    dots_img = cv2.circle(dots_img, (dst[2][0], dst[2][1]), 5, (0,255,0), -1)
    dots_img = cv2.circle(dots_img, (dst[3][0], dst[3][1]), 5, (0,255,0), -1)
    
    #save images
    if not os.path.exists("../output_images/points"):
        os.mkdir("../output_images/points")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/points", root + "-points" + ext)
    #save the result image
    mpimg.imsave(result_filename, dots_img, cmap='gray')