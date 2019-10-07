'''
Created on 25 sept. 2019

@author: cbraeuninger
'''
import glob
import matplotlib.image as mpimg
from Camera_Calibration import calibrate_camera
from Undistort_Image import undistort_image
from VisualizationHelpers import saveResultImage

#get calibration matrix
mtx, dst = calibrate_camera()

#read in all images with names with pattern calibration*.jpg
images = glob.glob('../camera_cal/calibration*.jpg')

# loop over images and undistort them
for file_name in images:
    # read in image
    img = mpimg.imread(file_name)
    # undistort image
    undist_img = undistort_image(img, mtx, dst)
    #save resulting image
    saveResultImage(undist_img, "../output_images/undistorted_cal_images", file_name, "-undist")
