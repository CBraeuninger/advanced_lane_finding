import cv2
import numpy as np
import math


def hough_lines_detection(img):
    '''
    Detects line segments in the lower part of the image
    (where the lane lines are),
    returns a collection of line segments each described by the coordinates
    of their end points
    '''

    # transform image to grayscale in order to be
    # able to feed it into the HoughLinesP function
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # mask image with trapezoid
    masked_image = lane_line_mask(gray)

    # -------------------------------------define parameters of Hough transform
    # this is the distance resolution of the accumulator in pixels
    rho = 1
    # this is the angular resolution of the accumulator in pixels
    theta = np.pi/180
    # Accumulator threshold parameter.
    # Only those lines are returned that get enough votes (>threshold)
    threshold = 7
    # Minimum line length. Line segments shorter than that are rejected.
    min_line_length = 25
    # Maximum allowed gap in between points considered to be on the same line.
    max_line_gap = 5

    # do Hough transform
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    return lines


def lane_line_mask(img):
    """
    Applies an image mask.
    Only keeps the region of the image where the lane lines can be.
    The rest of the image is set to black.
    """

    # ------------------------------------define vertices of region of interest
    # get dimensions of image
    ysize = img.shape[0]
    xsize = img.shape[1]

    # defining blank masks to start with
    mask = np.zeros_like(img)
    tmask = np.ones_like(img)*255

    # define vertices of trapezoid mask
    vertices = np.array([[(int(round(0.1*xsize)), ysize),
                        (int(round(0.45*xsize)), int(round(0.55*ysize))),
                        (int(round(0.55*xsize)), int(round(0.55*ysize))),
                        (int(round(0.9*xsize)), ysize)]],
                        dtype=np.int32)

    # defining a 3 channel or 1 channel color to fill the mask
    # with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        # declare tuples with value 255/0 and length channel_count
        keep_mask_color = (255,) * channel_count  # keep mask color is white
        remove_mask_color = (0,) * channel_count  # remove mask color is black
    else:
        keep_mask_color = 255
        remove_mask_color = 0

    # filling pixels inside the polygon defined
    # by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, keep_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    # Also mask triangle in the middle/lower third of the image
    tvertices = np.array([[(int(round(0.33*xsize)), ysize),
                           (int(round(0.5*xsize)), int(round(0.4*ysize))),
                           (int(round(0.66*xsize)), ysize)]], dtype=np.int32)
    # defining the mask
    cv2.fillPoly(tmask, tvertices, remove_mask_color)

    # mask color here has to be black (0,0,0) since we want to
    # remove those pixels from the image
    masked_image = cv2.bitwise_and(masked_image, tmask)

    return masked_image


def find_points(lines, img_width, img_height, line_fit):
    '''
    finds points on the lines as input for the perspective transformation
    and defines destination points
    '''
    # initialize variables to store the current maximum
    # line length (left and right)
    max_len_left = 0
    max_len_right = 0

    # intialize variables for the lines fitted to the left and right lane line
    left_line = None
    right_line = None

    # As source points take the endpoints of the longest
    # identified line segments on the left and the right

    if lines is not None:
        for line in lines:
            # each line segment consists of two points that
            # define the line, get their coordinates
            x1 = line[0][0]
            x2 = line[0][2]
            y1 = line[0][1]
            y2 = line[0][3]
            # calculate the slope of the line segment
            slope = (y2-y1)/(x2-x1)
            # calculate the length of the line segment
            length = math.sqrt((x1-x2)**2+(y1-y2)**2)

            # If the slope is negative the line segment belongs to the left
            # line (the origin is in the upper left corner)
            # if it is longer than the current champion,
            # set left_line to this line segment
            if slope <= 0 and length > max_len_left:
                left_line = line
                max_len_left = length
            # If the slope is positive,
            # the line segment belongs to the right line
            elif slope > 0 and length > max_len_right:
                right_line = line
                max_len_right = length

    # if at least one line was found on the left side
    if left_line is not None:
        # Fit 1D polynomial to the line: y = mx + B
        line_fit.set_l_fit(np.polyfit([left_line[0][1], left_line[0][3]],
                                      [left_line[0][0], left_line[0][2]], 1))

    # make a function
    left_line_func = np.poly1d([line_fit.get_l_fit()[0],
                                line_fit.get_l_fit()[1]])

    # same for the right line
    if right_line is not None:
        line_fit.set_r_fit(np.polyfit([right_line[0][1], right_line[0][3]],
                                      [right_line[0][0], right_line[0][2]], 1))

    right_line_func = np.poly1d([line_fit.get_r_fit()[0],
                                 line_fit.get_r_fit()[1]])

    # Source points:
    # 1: value of left_line_fit at bottom of image
    # 2: value of left_line_fit at 20% of image from bottom
    # 3: value of right_line_fit at 20% of image from bottom
    # 4: value of right_line_fit at bottom of image
    yBottom = img_height
    yTop = 0.8*img_height
    src = np.float32([[left_line_func(yBottom), yBottom],
                      [left_line_func(yTop), yTop],
                      [right_line_func(yTop), yTop],
                      [right_line_func(yBottom), yBottom]])

    # destination points:
    # 1: 20% from left border, at bottom
    # 2: same x-value, at top
    # 3: 80% from left border, at top
    # 4: same x-value, at bottom
    dst = np.float32([[0.2*img_width, img_height], [0.2*img_width, 0],
                      [0.8*img_width, 0], [0.8*img_width, img_height]])

    return src, dst


def warp_image(img, src, dst):
    '''
    Applies a perspective transformation on an image
    using source and destination points
    '''

    # get the transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # warp image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    return warped


def do_perspective_transform(img, line_fit):

    '''
    Detects source and destination points from a grayscale version of an image
    (using Hough transform) and then does the perspective transformation of the
    image
    '''

    # get the lane lines
    lines = hough_lines_detection(img)

    # Calculate source and destination points
    src, dst = find_points(lines, img.shape[1], img.shape[0], line_fit)

    # mask image
    masked = lane_line_mask(img)

    # warp the image
    warped = warp_image(masked, src, dst)

    return warped, src, dst
