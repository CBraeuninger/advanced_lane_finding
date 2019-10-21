from curvature import real_lane_curvature, distance_from_lane, get_conversion
import cv2
import numpy as np
from globals import config


def add_lane_lines(img, lane_line_img):
    """
    Colors lane lines in an image
    """
    ret_img = img.copy()
    ret_img[(lane_line_img[:, :, 0] > 0)] = (255, 0, 0)
    ret_img[(lane_line_img[:, :, 2] > 0)] = (0, 0, 255)

    return ret_img


def add_info(img, yeval, leftx, lefty, rightx, righty, l_left_seg, l_right_seg,
             leftx_base, rightx_base, src):
    """
    Adds info about curvature and departure from the middle of the lane
    """

    # Get conversion factor
    ym_per_pix, xm_per_pix = get_conversion(img.shape[0], leftx_base,
                                            rightx_base)

    # Calculate the curvature and append it to the array where
    # we save the last 100 values for averaging
    config.curvature.append_array(real_lane_curvature(yeval, leftx, lefty,
                                                      rightx, righty,
                                                      l_left_seg, l_right_seg,
                                                      ym_per_pix, xm_per_pix),
                                  100)

    # Calculate the average value of the curvature over the last 100 frames
    avg_curvature = config.curvature.average()

    # Calculate the distance from the middle of the lane and append it
    # to the array where we save the last 100 values for averaging
    config.offset.append_array(distance_from_lane(img, src), 100)

    # Calculate the average value of the distance from the middle of the lane
    avg_offset = config.offset.average()

    if avg_offset > 0:
        side = "left"
    else:
        side = "right"

    # Add the info about the curvature to the image
    img = cv2.putText(img,
                      "Curvature radius in m: {0:.2f} m".format(avg_curvature),
                      (50, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                      (255, 255, 255), bottomLeftOrigin=False)
    # Add the info about the distance from the middle of the lane to the image
    img = cv2.putText(img, "Vehicle is {0:.2f} m ".format(abs(avg_offset)) +
                      side + " of center.", (50, 100), cv2.FONT_HERSHEY_PLAIN,
                      2, (255, 255, 255), bottomLeftOrigin=False)

    return img


def add_trapezoid(img, src):
    """
    Returns an image of a green trapezoid with the source points of the
    perspective transform as vertices
    """
    # Vertices of the trapezoid shall be the detected source points for the
    # perspective transformation of the image
    points = np.array([[(src[0][0], src[0][1]), (src[1][0], src[1][1]),
                        (src[2][0], src[2][1]), (src[3][0], src[3][1])]],
                      dtype=np.int32)

    img = cv2.fillPoly(img, points, (0, 255, 0))

    return img


def final_image(img, lane_line_img, yeval, leftx, lefty, rightx, righty,
                l_left_seg, l_right_seg, src, leftx_base, rightx_base):
    """
    Constructs the final image that will replace each frame in the video
    """
    # Color the lane line pixels
    img = add_lane_lines(img, lane_line_img)

    # Add info about curvature and departure from the middle of the lane
    img = add_info(img, yeval, leftx, lefty, rightx, righty,
                   l_left_seg, l_right_seg, leftx_base, rightx_base, src)

    # Get an image of a trapezoid with source points of perspective
    # transformation as vertices
    trap_img = add_trapezoid(np.zeros_like(img), src)

    # Add trapezoid to final image
    res_img = cv2.addWeighted(img, 1.0, trap_img, 0.5, 0.0)

    return res_img
