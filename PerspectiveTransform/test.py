'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from PerspectiveTransform import doPerspectiveTransform
from VisualizationHelpers import saveResultImage


#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:
    
    img = cv2.imread(file_name)

    warped, src, dst = doPerspectiveTransform(img)
    #save images
    saveResultImage(warped, "../output_images/warped", file_name, "-warped", True)
    