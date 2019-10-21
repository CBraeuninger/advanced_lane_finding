from gradient import combine_gradient
from hls_select import saturation_select
import cv2
import numpy as np


def create_binary_image(img):

    '''
    Creates the binary image for line detection
    output is an rgb image with only black and white pixels,
    not a "real" binary image
    '''

    # ------------------------------------------------------ get gradient image
    grad = combine_gradient(img)

    # ----------------- get saturation image and combine it with gradient image
    return cv2.bitwise_or(grad, saturation_select(img, (120, 255), 120, 'rgb'))
