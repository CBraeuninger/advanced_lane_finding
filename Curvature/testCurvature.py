'''
Created on 8 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from Curvature import realLaneCurvature

#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/final/*.jpg')

#open file to write curvatures
file_obj = open("Curvatures.txt", 'w')

# loop over images
for file_name in images:
    
    img = cv2.imread(file_name)
    
    #evaluate curvature at the bottom of the image (closest to the vehicle)
    curvature = realLaneCurvature(img, img.shape[0])
    
    file_obj.write("Image: " + file_name + ", Curvature radius: '{0}' m\n".format(curvature))

file_obj.close()