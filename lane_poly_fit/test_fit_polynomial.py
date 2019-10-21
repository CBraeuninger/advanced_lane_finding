import glob
import cv2
from lane_poly_fit import fit_polynomial, find_lane_pixels,\
    find_starting_points
from visualization import save_result_image
from globals.fits import Fit

# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/warped/*.jpg')

# loop over images
for file_name in images:

    img = cv2.imread(file_name)

    # get starting points of pixel detection
    leftx_base, rightx_base = find_starting_points(img)
    # find lane pixels
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, img = \
        find_lane_pixels(img, leftx_base, rightx_base, True)

    fit = Fit()
    left_fit, right_fit, out_img =\
        fit_polynomial(leftx, lefty, rightx, righty, fit, True, img)

    save_result_image(out_img, "../output_images/polynomial", file_name,
                      "-poly", True)
