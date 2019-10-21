import cv2
import glob
from perspective_transform import hough_lines_detection
from visualization import save_result_image


# import images
# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    # read in example image
    img = cv2.imread(file_name)
    # detect lines using Hough algorithm
    lines = hough_lines_detection(img)

    # loop over all the lines and draw them on the image
    for line in lines:
        img = cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]),
                       (255, 0, 0), 5)

    # save images
    save_result_image(img, "../output_images/HoughLines", file_name, "-hough",
                      True)
