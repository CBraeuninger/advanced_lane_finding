'''
Created on 20 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from HLS_select import color_select
from VisualizationHelpers import saveResultImage

#import images
#read in all images with names with pattern *.jpg
images = glob.glob('../output_images/undistorted_images/*.jpg')

# loop over images and undistort them
for file_name in images:
    
    img = cv2.imread(file_name)
    yellow = color_select(img, (20,35), 120, 120)
    white = color_select(img, (0,180), 240, 120)
    
    hls = cv2.bitwise_or(yellow, white)
    #save images
    saveResultImage(hls, "../output_images/hls_color", file_name, "-hls_color", True)