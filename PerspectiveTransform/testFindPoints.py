'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from PerspectiveTransform import houghLinesDetection, findPoints
from VisualizationHelpers import drawDots, saveResultImage, addText

#import images
#read in all binary images with names with pattern *.jpg (output of hls selection)
images = glob.glob('../output_images/hls/*.jpg')

# loop over images
for file_name in images:

    #read in example image
    img = cv2.imread(file_name)
    gray = cv2.imread(file_name,0)
    
    #do Hough transform
    lines = houghLinesDetection(img)
    #get source and destination points
    src, dst = findPoints(lines, img.shape[1], img.shape[0])
        
    #Transform image to RGB
    dots_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    #draw dot (small circle on image)
    #source points
    for i in range(4):
        dots_img = drawDots(dots_img, (src[i][0], src[i][1]))
        dots_img = addText(dots_img, "["+str(i)+"]", (src[i][0], src[i][1]))
    
    #destination points
    for i in range(4):
        dots_img = drawDots(dots_img, (dst[i][0], dst[i][1]), (0,255,0))
        ots_img = addText(dots_img, "["+str(i)+"]", (dst[i][0], dst[i][1]), (0,255,0))
    
    #save images
    saveResultImage(dots_img, "../output_images/points", file_name, "-points", True)
    