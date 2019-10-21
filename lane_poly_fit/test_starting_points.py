import cv2
import glob
from lane_poly_fit import find_starting_points
from visualization import draw_dots, save_result_image

# import images
# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/warped/*.jpg')

# loop over images
for file_name in images:

    # read in example image
    img = cv2.imread(file_name)

    leftx_base, right_xbase = find_starting_points(img)

    points = draw_dots(img, (leftx_base, img.shape[0]-5))
    points = draw_dots(points, (right_xbase, img.shape[0]-5), (0, 0, 255))

    save_result_image(points, "../output_images/starting_points", file_name,
                      "-start", True)
