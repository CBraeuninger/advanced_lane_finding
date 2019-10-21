import glob
import cv2
from perspective_transform import hough_lines_detection, find_points
from visualization import draw_dots, save_result_image, add_text
from globals.fits import LineFit

# import images
# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/hls/*.jpg')

# loop over images
for file_name in images:

    # read in example image
    img = cv2.imread(file_name)
    gray = cv2.imread(file_name, 0)

    # initialize line_fit
    line_fit = LineFit(img.shape[1], img.shape[0])

    # do Hough transform
    lines = hough_lines_detection(img)
    # get source and destination points
    src, dst = find_points(lines, img.shape[1], img.shape[0], line_fit)

    # Transform image to RGB
    dots_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # draw dot (small circle on image)
    # source points
    for i in range(4):
        dots_img = draw_dots(dots_img, (src[i][0], src[i][1]))
        dots_img = add_text(dots_img, "["+str(i)+"]", (src[i][0], src[i][1]))

    # destination points
    for i in range(4):
        dots_img = draw_dots(dots_img, (dst[i][0], dst[i][1]), (0, 255, 0))
        y_offset = 0 if i == 0 or i == 3 else 20
        dots_img = add_text(dots_img, "["+str(i)+"]",
                            (dst[i][0], int(dst[i][1]+y_offset)),
                            color=(0, 255, 0))

    # save images
    save_result_image(dots_img, "../output_images/points", file_name,
                      "-points", True)
