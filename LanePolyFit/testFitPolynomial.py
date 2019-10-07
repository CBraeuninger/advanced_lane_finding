'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from LanePolyFit import fitPolynomial
from VisualizationHelpers import saveResultImage

#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/final/*.jpg')

# loop over images
for file_name in images:
    
    img = cv2.imread(file_name)

    left_fit, right_fit, out_img = fitPolynomial(img, True)
    
    saveResultImage(out_img, "../output_images/polynomial", file_name, "-poly", True)