'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import cv2
import glob
from PerspectiveTransform import houghLinesDetection
from VisualizationHelpers import saveResultImage


#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    #read in example image
    img = cv2.imread(file_name)
    
    #detect lines using Hough algorithm
    lines = houghLinesDetection(img)
    
    #loop over all the lines and draw them on the image
    for line in lines:
        img = cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255,0,0), 5)
        
    #save images
    saveResultImage(img, "../output_images/HoughLines", file_name, "-hough", True)
