import glob
import cv2
from curvature import real_lane_curvature, get_conversion
from lane_poly_fit import find_lane_pixels, find_starting_points

# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/warped/*.jpg')

# open file to write curvatures
file_obj = open("Curvatures.txt", 'w')

# loop over images
for file_name in images:

    img = cv2.imread(file_name)

    # get starting points of pixel detection
    leftx_base, rightx_base = find_starting_points(img)

    # detect lane line pixels
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, out_img =\
        find_lane_pixels(img, leftx_base, rightx_base)

    # calculate conversion factors
    xm_per_pix, ym_per_pix = get_conversion(img.shape[0], leftx_base,
                                            rightx_base)

    # evaluate curvature at the bottom of the image (closest to the vehicle)
    curv = real_lane_curvature(img.shape[0], leftx, lefty, rightx, righty,
                               l_left_seg, l_right_seg, xm_per_pix, ym_per_pix)

    file_obj.write("Image: " + file_name +
                   ", Average radius: '{0}' m\n".format(curv))

file_obj.close()
