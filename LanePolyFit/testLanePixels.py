'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from LanePolyFit import findLanePixels, findStartingPoints
from VisualizationHelpers import saveResultImage

#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/final/*.jpg')

# loop over images
for file_name in images:
    
    img = cv2.imread(file_name)

    #find starting points
    leftx_base, rightx_base = findStartingPoints(img)
    
    #find the lane pixels
    leftx, lefty, rightx, righty, out_img = findLanePixels(img, leftx_base, rightx_base, True)
    
    saveResultImage(out_img, "../output_images/windows", file_name, "-windows", True)