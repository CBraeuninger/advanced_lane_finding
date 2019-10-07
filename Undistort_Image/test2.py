'''
Created on 6 oct. 2019

@author: cbraeuninger
'''
import glob
import matplotlib.image as mpimg
import os
from Camera_Calibration import calibrate_camera
from Undistort_Image import undistort_image

#get calibration matrix
mtx, dst = calibrate_camera()

#read in all images with names with pattern calibration*.jpg
images = glob.glob('../test_images/*.jpg')

# loop over images and undistort them
for file_name in images:
    # read in image
    img = mpimg.imread(file_name)
    # undistort image
    undist_img = undistort_image(img, mtx, dst)
    #save images
    if not os.path.exists("../output_images/undistorted_images"):
        os.mkdir("../output_images/undistorted_images")
    #get filename of result image
    (head, tail) = os.path.split(file_name)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join("../output_images/undistorted_images", root + "-undist" + ext)

    #save the result image
    mpimg.imsave(result_filename, undist_img)