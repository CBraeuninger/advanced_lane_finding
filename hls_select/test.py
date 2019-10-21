'''
Created on 1 oct. 2019

@author: cbraeuninger
'''
import glob
import cv2
from hls_select import saturation_select
from visualization import save_result_image

# import images
# read in all images with names with pattern *.jpg
images = glob.glob('../output_images/undistorted_images/*.jpg')

# loop over images and undistort them
for file_name in images:

    img = cv2.imread(file_name)
    hls = saturation_select(img, (185, 255), 120)
    # save images
    save_result_image(hls, "../output_images/hls", file_name, "-hls", True)
