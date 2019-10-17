'''
Created on 17 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from PerspectiveTransform import trapezoidMask
from VisualizationHelpers import saveResultImage

#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    #read in example image
    img = cv2.imread(file_name)
    
    #mask the image
    masked = trapezoidMask(img)
        
    #save images
    saveResultImage(masked, "../output_images/Masked", file_name, "-masked", True)
