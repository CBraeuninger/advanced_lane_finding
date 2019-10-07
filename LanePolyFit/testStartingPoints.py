'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
import cv2
import glob
from LanePolyFit import findStartingPoints
from VisualizationHelpers import drawDots, saveResultImage

#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/final/*.jpg')

# loop over images
for file_name in images:

    #read in example image
    img = cv2.imread(file_name)

    leftx_base, right_xbase = findStartingPoints(img)
    
    points = drawDots(img, (leftx_base,img.shape[0]-5))
    points = drawDots(points, (right_xbase, img.shape[0]-5), (0,0,255))
    
    saveResultImage(points, "../output_images/starting_points", file_name, "-start", True)