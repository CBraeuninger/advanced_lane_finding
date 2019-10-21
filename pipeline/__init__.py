import glob
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
from camera_calibration import calibrate_camera
from undistort import undistort_image
from hls_select import saturation_select
from perspective_transform import do_perspective_transform, warp_image
from visualization import save_result_image
from lane_poly_fit import find_lane_pixels, color_lane_pixels,\
    find_starting_points
from final_image import final_image
from binary_image import create_binary_image
from globals.fits import Fit


# ------------------------------------------------------ first calibrate camera
# get camera matrix and distortion coefficients
mtx, dist_coeff = calibrate_camera()

# --------------------------------------------------------------- import images

# read in all images with names with pattern *.jpg
images = glob.glob('../test_images/*.jpg')

# loop over images
for file_name in images:

    img = mpimg.imread(file_name)

    # --------------------------------------------------------- undistort image
    undist = undistort_image(img, mtx, dist_coeff)

    # - convert to hls color space and apply threshold to generate binary image
    binary = create_binary_image(undist)

    # ------------------------------------- transform to bird's eye perspective
    warped, src, dst = do_perspective_transform(binary, Fit())

    # ---------------------------------------------------------- fit polynomial
    leftx_base, rightx_base = find_starting_points(warped)
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, output_img =\
        find_lane_pixels(warped, leftx_base, rightx_base)

    # ------------------------------ color detected lane pixels in warped image
    lane_pix = color_lane_pixels(np.zeros_like(warped), leftx, lefty,
                                 rightx, righty)

    # ------------------------------------------------------------ unwarp image
    lane_pix_unwarped = warp_image(lane_pix, dst, src)

    # --------- superpose image of unwarped lane pixels image on original image
    # ------------------------ and add curvature and offset from middle of lane
    res_img = final_image(img, lane_pix_unwarped, img.shape[0], leftx, lefty,
                          rightx, righty, l_left_seg, l_right_seg, src,
                          leftx_base, rightx_base)

    # ------------------------------------------------------------- save images
    save_result_image(res_img, "../output_images/final", file_name, "-final",
                      True)
