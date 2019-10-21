import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from globals import config


def find_starting_points(img):
    '''
    finds the starting points of the polynomial to fit to the lines
    '''
    # take a histogram of the image
    histogram = np.sum(img[:, :, 0], axis=0)
    # find midpoint of histogram
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base


def find_lane_pixels(img, leftx_base, rightx_base, visualize=False):
    """
    Finds pixels belonging to the lane lines by searching in a window around
    the starting points. Once enough pixels have been found in one window, the
    search window is shifted towards the top of the image
    """
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Take only one channel of RGB image
    # and divide by 255 to make a binary image
    binary_warped = img[:, :, 0]/255

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # initialize variables that will give us an indication
    # of the length of the detected lane segment
    l_left_seg = 0
    l_right_seg = 0

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # if windows shall be visualized on output image, draw them
        if visualize:
            # Draw the windows on the visualization image
            cv2.rectangle(img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high),
                          (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the left window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]

        # if lane pixels were found in the window increase l_left_seg
        # (indicator of the length of the left lane line)
        if not good_left_inds.size == 0:
            l_left_seg += 1

        # Identify the nonzero pixels in x and y within the right window
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # if lane pixels were found in the window increase l_right_seg
        # (indicator of the length of the right lane line)
        if not good_right_inds.size == 0:
            l_right_seg += 1

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window
        left_window = binary_warped[win_y_low:win_y_high,
                                    win_xleft_low:win_xleft_high]
        left_histogram = np.sum(left_window, axis=0)
        if (sum(left_histogram) > minpix):
            leftx_current = np.argmax(left_histogram) + win_xleft_low

        right_window = binary_warped[win_y_low:win_y_high,
                                     win_xright_low:win_xright_high]
        right_histogram = np.sum(right_window, axis=0)
        if (sum(right_histogram) > minpix):
            rightx_current = np.argmax(right_histogram) + win_xright_low

    # Concatenate the arrays of indices
    # (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    output_img = img.copy()

    # color pixels on the left lane line red
    output_img[lefty, leftx] = [255, 0, 0]
    # color pixels on the right lane line blue
    output_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, l_left_seg, l_right_seg, output_img


def fit_polynomial(leftx, lefty, rightx, righty, fit, visualize=False,
                   img=np.array([], [])):
    """
    Fits a 2nd order polynomial to the detected pixels on the lane lines
    """
    # Fit a poltnomial x = A*y**2 + B*y + C to the detected lane line pixels
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except TypeError:
        print('The function failed to fit a line!')
        # if no polynomial could be fit, conserve the old one
        # (from the previous frame)
        left_fit = fit.get_l_fit()

    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        print('The function failed to fit a line!')
        # if no polynomial could be fit, conserve the old one
        # (from the previous frame)
        right_fit = fit.get_r_fit()

    # if result shall be visulized, draw polynomial on output image
    # and color the detected lane pixels
    if visualize:
        # Generate y values for plotting
        lefty = np.linspace(0, img.shape[0]-1, img.shape[0], dtype='int')
        righty = np.linspace(0, img.shape[0]-1, img.shape[0], dtype='int')

        # The x values corresponding to the y values can be calculated using
        # the polynomial
        left_fitx = (left_fit[0]*lefty**2 + left_fit[1]*lefty +
                     left_fit[2]).astype(int)
        right_fitx = (right_fit[0]*righty**2 + right_fit[1]*righty +
                      right_fit[2]).astype(int)

        # remove points that are outside the image boundaries

        ind_to_delete = []

        for i in range(lefty.size):
            if left_fitx[i] >= img.shape[1]:
                ind_to_delete.append(i)

        lefty = np.delete(lefty, ind_to_delete)
        left_fitx = np.delete(left_fitx, ind_to_delete)

        ind_to_delete = []

        for i in range(righty.size):
            if right_fitx[i] >= img.shape[1]:
                ind_to_delete.append(i)

        righty = np.delete(righty, ind_to_delete)
        right_fitx = np.delete(right_fitx, ind_to_delete)

        # Plots the left and right polynomials on the lane lines
        img[lefty, left_fitx] = [255, 255, 0]
        img[righty, right_fitx] = [255, 255, 0]

    return left_fit, right_fit, img


def search_around_poly(img, left_fit, right_fit):
    """
    Searches for lane line pixels around the polynomial fitted to the
    lane line pixels detected in the previous frame
    """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 50

    # Take only one channel of RGB image and divide by 255
    # to make a binary image
    binary_warped = img[:, :, 0]/255

    # get image width
    img_wid = binary_warped.shape[1]

    # Take left side of image
    left_bw = binary_warped[:, :int(0.5*binary_warped.shape[1])]
    right_bw = binary_warped[:, int(0.5*binary_warped.shape[1]):]

    # Grab activated pixels
    # We must do this seperately for left and right side of the image to avoid
    # the pixel identification to "tilt" to one side

    left_nonzero = left_bw.nonzero()
    left_nonzeroy = np.array(left_nonzero[0])
    left_nonzerox = np.array(left_nonzero[1])

    right_nonzero = right_bw.nonzero()
    right_nonzeroy = np.array(right_nonzero[0])
    right_nonzerox = np.array(right_nonzero[1])

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_fit_func = np.poly1d(left_fit)
    right_fit_func = np.poly1d(right_fit)
    left_lane_inds = ((left_nonzerox >= left_fit_func(left_nonzeroy)-margin) &
                      (left_nonzerox < left_fit_func(left_nonzeroy)+margin)
                      ).nonzero()[0]

    right_lane_inds = ((right_nonzerox+0.5*img_wid >=
                        right_fit_func(right_nonzeroy)-margin) &
                       (right_nonzerox+0.5*img_wid <
                        right_fit_func(right_nonzeroy)+margin)
                       ).nonzero()[0]

    # extract left and right line pixel positions
    leftx = left_nonzerox[left_lane_inds]
    lefty = left_nonzeroy[left_lane_inds]
    rightx = right_nonzerox[right_lane_inds] + int(0.5*img_wid)
    righty = right_nonzeroy[right_lane_inds]

    # get an estimate which lane line is longer
    if (leftx == 0).all() or (lefty == 0).all():
        l_left_seg = 0
    else:
        l_left_seg = math.sqrt((leftx.max()-leftx.min())**2 +
                               (lefty.max()-lefty.min())**2)
    if (rightx == 0).all() or (righty == 0).all():
        l_right_seg = 0
    else:
        l_right_seg = math.sqrt((rightx.max()-rightx.min())**2 +
                                (righty.max()-righty.min())**2)

    return leftx, lefty, rightx, righty, l_left_seg, l_right_seg


def find_lanes(img):
    """
    Finds the pixels belonging to the lane lines
    Also gives back an indicator on wether the right or the left
    detected line segment is longer
    """
    # find starting points
    leftx_base, rightx_base = find_starting_points(img)

    # if no polynomial was fitted yet to the left or right lane line
    # search in windows starting at leftx_base and rightx_base
    if (config.fit.get_l_fit() == np.array([0, 0, 0])).all() or\
            (config.fit.get_r_fit() == np.array([0, 0, 0])).all():

        leftx, lefty, rightx, righty, l_left_seg, l_right_seg, _ =\
            find_lane_pixels(img, leftx_base, rightx_base, False)

    # if polynomials were already fitted to the left and right lane lines,
    # search in windows around them
    else:
        leftx, lefty, rightx, righty, l_left_seg, l_right_seg =\
            search_around_poly(img, config.fit.get_l_fit(),
                               config.fit.get_r_fit())
    # fit polynomials to the detected points
    left_fit, right_fit, img = fit_polynomial(leftx, lefty, rightx, righty,
                                              config.fit)
    # write polynomials to global variables in order to be able to
    # reuse them in the next frame
    config.fit.set_l_fit(left_fit)
    config.fit.set_r_fit(right_fit)

    return leftx, lefty, rightx, righty, l_left_seg, l_right_seg, leftx_base,\
        rightx_base


def color_lane_pixels(img, leftx, lefty, rightx, righty):
    """
    returns an image where the detected lane pixels are
    colored
    """
    # copy image so as not to alter the original
    pix_img = img.copy()

    # color image red where the left lane was detected
    pix_img[lefty, leftx] = [255, 0, 0]

    # blue for the right lane line
    pix_img[righty, rightx] = [0, 0, 255]

    return pix_img
