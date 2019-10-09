'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from LanePolyFit import fitPolynomial, findLanePixels
from VisualizationHelpers import saveResultImage

#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/warped/*.jpg')

# loop over images
for file_name in images:
    
    img = cv2.imread(file_name)
    
    #find lane pixels
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, img = findLanePixels(img, True)

    left_fit, right_fit, out_img = fitPolynomial(leftx, lefty, rightx, righty, True, img)
    
    saveResultImage(out_img, "../output_images/polynomial", file_name, "-poly", True)