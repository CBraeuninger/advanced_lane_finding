'''
Created on 6 oct. 2019

@author: cbraeuninger
'''

import cv2
import numpy as np
import math
from prompt_toolkit.layout.processors import Transformation

def houghLinesDetection(img):
    '''
    Detect line segment in the lower part of the image (where the lane lines are)
    Returns a collection of line segments each described by the coordinates of their end points
    '''
    
    #take lower half of the image
#     half_size = int(round(0.5*img.shape[0]))
#     img[:,:half_size] = 0

    #mask image with trapezoid
    masked_image = trapezoidMask(img)    
    
    #define parameters of Hough transform
    rho = 1 #this is the distance resolution of the accumulator in pixels
    theta = np.pi/180 #this is the angular resolution of the accumulator in pixels
    threshold = 7 #Accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold)
    min_line_length = 25 #Minimum line length. Line segments shorter than that are rejected.
    max_line_gap = 1 #Maximum allowed gap in between points considered to be on the same line.
    
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    return lines



def trapezoidMask(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #define vertices of region of interest
    ysize = img.shape[0]
    xsize = img.shape[1]
    
    vertices = np.array([[(int(round(0.1*xsize)), ysize),\
                        (int(round(0.45*xsize)), int(round(0.55*ysize))),\
                        (int(round(0.55*xsize)), int(round(0.55*ysize))),\
                        (int(round(0.9*xsize)), ysize)]],\
                        dtype=np.int32) 
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count #declare tuple with value 255 and length channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image



def findPoints(lines, img_width, img_height):
    '''
    finds points on the lines as input for the perspective Transformation and defines destination points
    '''
    #initialize variables to store the current maximum line length (left and right)
    max_len_left = 0
    max_len_right = 0
    
    if lines is not None:
        for line in lines:
            #each line segment consists of two points that define the line, get their coordinates
            x1 = line[0][0]
            x2 = line[0][2]
            y1 = line[0][1]
            y2 = line[0][3]
            #calculate the slope of the line segment
            slope = (y2-y1)/(x2-x1)
            #calculate the length of the line segment
            length = math.sqrt((x1-x2)**2+(y1-y2)**2)
            
            #If the slope is negative the line segment belongs to the left line
            #(the origin is in the upper left corner)
            #if it is longer than the current champion, set left_line to this line segment
            if slope < 0 and length>max_len_left:
                left_line = line
                max_len_left = length
            #If the slope is positive, the line segment belongs to the right line
            elif slope > 0 and length>max_len_right:
                right_line = line
                max_len_right = length
    
    #The four source points are the end points of the selected left and right line segments
    src =  np.float32([[left_line[0][0], left_line[0][1]], [left_line[0][2], left_line[0][3]],\
                       [right_line[0][0], right_line[0][1]], [right_line[0][2], right_line[0][3]]])
    
    #First destination point shall be at x = 10% image width and y = 10% image height
    x1 = 0.1*img_width
    y1 = 0.1*img_height
    
    #Second point: Same x-coordinate and y-coordinate = y1 + max_len_left
    x2 = x1
    y2 = y1 + max_len_left
    
    #Third point (on right lane): x-coordinate = x1 + x-distance point 1 left lane - point 1 right lane
    #y-coordinate = y1 + y-distance point 1 left lane - point 1 right lane
    x3 = x1 + abs(right_line[0][0] - left_line[0][0])
    y3 = y1 + abs(right_line[0][1] - left_line[0][1])
    
    #Fourth point: x-coordinate = x3, y-coordinate = y3 + max_len_right
    x4 = x3
    y4 = y3 + max_len_right
    
    dst = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    
    return src, dst

def warpImage(img, src, dst):
    '''
    Applies a perspective transformation on an image using source and destination points
    '''
    
    #get the transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    #warp image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped

def doPerspectiveTransform(img):
    
    '''
    Detects source and destination points and the does the perspective transformation on a binary image
    '''
    #get the lane lines
    lines = houghLinesDetection(img)
    
    #Calculate source and destination points
    src, dst = findPoints(lines, img.shape[1], img.shape[0])
    
    #warp the image
    warped = warpImage(img, src, dst)
    
    return warped
    
        