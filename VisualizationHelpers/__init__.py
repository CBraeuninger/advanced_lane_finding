'''
Created on 7 oct. 2019

@author: cbraeuninger
'''
import cv2
import os
import matplotlib.image as mpimg

from bleach._vendor.html5lib._ihatexml import name

def drawDots(dots_img, coordinates, color=(255,0,0), size = 5):
    '''Draws dots at the coordinates on image
    '''
    dots_img = cv2.circle(dots_img, (coordinates[0], coordinates[1]), size, color, -1)
    
    return dots_img

def saveResultImage(img, output_path, old_filename, suffix, isGrayScale=False):
    '''
    saves image to directory, attaching suffix to its name
    '''
    #save images
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #get filename of result image
    headTail = os.path.split(old_filename)
    (root, ext) = os.path.splitext(headTail[1])
    result_filename = os.path.join(output_path, root + suffix + ext)

    #save the result image
    if isGrayScale:
        mpimg.imsave(result_filename, img, cmap='gray')
    else:
        mpimg.imsave(result_filename, img)