from lane_poly_fit import fit_polynomial
from globals import config


def get_conversion(img_height, leftx_base, rightx_base):
    """
    Get the conversion factors for pixel/meters
    """
    # y-direction:
    # the lane lines are about 30 m long
    # in the perspective transform we take about half of that
    # and project it to the warped image
    ym_per_pix = 15/img_height

    # the lane is about 3.7 m wide
    # in the warped image that corresponds to the number of pixels between
    # the left and right lane
    xm_per_pix = 3.7/(rightx_base - leftx_base)

    return ym_per_pix, xm_per_pix


def calculate_curvature(fit, yeval, ym_per_pix):
    """
    Calculate the curvature of the road given a fit to one
    of the lane lines
    """
    # Get the parameters of the fit
    A = fit[0]
    B = fit[1]
    # calculate the curvature
    curv = (1+(2*A*yeval*ym_per_pix+B)**2)**1.5/abs(2*A)

    return curv


def real_lane_curvature(yeval, leftx, lefty, rightx, righty, l_left_seg,
                        l_right_seg, ym_per_pix, xm_per_pix):
    """
    Fits a polynomial to the lane lines (in real world space) and
    calculates the curvature of the road
    """
    # fit polynomial
    left_fit, right_fit, _ = fit_polynomial(leftx*xm_per_pix,
                                            lefty*ym_per_pix,
                                            rightx*xm_per_pix,
                                            righty*ym_per_pix,
                                            config.fit_real)

    # choose longer line segment and caculate curvature
    if l_right_seg > l_left_seg:
        curv = calculate_curvature(right_fit, yeval, ym_per_pix)
    else:
        curv = calculate_curvature(left_fit, yeval, ym_per_pix)
    # store fitted polynomials in global variables for use in
    # frames where no polynomial can be fitted
    config.fit_real.set_l_fit(left_fit)
    config.fit_real.set_r_fit(right_fit)

    return curv


def distance_from_lane(img, src):
    """
    Calculates the offset of the middle of the car (assumed
    to coincide with the middle of the camera image) and the
    middle of the lane line
    """
    # conversion factor for unwarped image
    #
    xm_per_pix = 3.7/(src[3][0]-src[0][0])

    # center of vehicle is at center of image
    x_veh = img.shape[1]/2*xm_per_pix

    # calculate middle of lane line
    x_middle = (0.5*src[0][0] + 0.5*src[3][0])*xm_per_pix

    # calculate offset
    offset = x_middle - x_veh

    return offset
