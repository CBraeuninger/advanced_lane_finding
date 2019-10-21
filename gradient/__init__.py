import cv2
import numpy as np


def abs_sobel_threshold(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    """
    Applies a threshold on the absolute value of the gradient in
    direction given by 'orient' (either x or y)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif (orient == 'y'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Applies a threshold on the magnitude to the gradient taken in
    x- and y-direction.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255*magnitude/np.max(magnitude))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Applies a threshold on the direction of the gradient
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx)
    # to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def combine_gradient(img, sobel_kernel=3, x_thresh=(100, 255),
                     y_thresh=(100, 255), mag_thresh=(50, 255),
                     dir_thresh=(np.pi/6, np.pi/3)):
    """
    Combine the gradient functions to generate a binary image.
    Pixels that are one in the binary image shall have magnitude of
    gradient in between 50 and 255 (max value) AND direction of gradient
    in between pi/6 and pi/3 rad (corresponds to 30° - 60°) OR -pi/3 to
    -pi/6
    OR
    gradient in x-direction between 100 and 255 AND gradient in y-direction
    between 50 and 255
    output is an rgb image with only black and white pixels,
    not a "real" binary image
    """

    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(img, sobel_kernel, 'x', x_thresh)
    grady = abs_sobel_threshold(img, sobel_kernel, 'y', y_thresh)
    mag_binary = mag_threshold(img, sobel_kernel, mag_thresh)
    dir_binary_pos = dir_threshold(img, sobel_kernel, dir_thresh)
    dir_binary_neg = dir_threshold(img, sobel_kernel,
                                   (-dir_thresh[1], -dir_thresh[0]))

    combined = np.zeros_like(mag_binary)
    combined[(((dir_binary_pos == 1) | (dir_binary_neg == 1))
              & (mag_binary == 1)) |
             ((gradx == 1) & (grady == 1))] = 1

    combined_rgb = np.zeros_like(img, dtype=np.uint8)
    combined_rgb[:, :, 0] = combined*255
    combined_rgb[:, :, 1] = combined*255
    combined_rgb[:, :, 2] = combined*255

    return combined_rgb
