import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import os
from numpy.testing._private.utils import suppress_warnings


def calibrate_camera(save_images_flag=False):

    # Arrays to store the object points and image
    # points for all calibration images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # The chess board has 9*6 inner points
    # mgrid[0:9,0:6] returns an array ([[[0,0,0,0,0,0], [1,1,1,1,1,1], ...,
    # [8,8,8,8,8,8]], [[0,1,2,3,4,5], ...[0,1,2,3,4,5]]])
    # Transpose returns an array ([[[0 0], [1 0], ..., [8,0]], [[0 1], [1 1],
    # ...[8,1]], ..., [[0 5], ..., [8 5]]])
    # Then reshape it to three columns and inferred number of rows
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # read in all images with names with pattern calibration*.jpg
    cal_images = glob.glob('../camera_cal/calibration*.jpg')

    # loop over calibration images to compute camera matrix
    # and distortion coefficients
    for file_name in cal_images:
        # read in image
        img = mpimg.imread(file_name)
        # convert image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # if corners are found, add object points and image points
        if ret is True:
            imgpoints.append(corners)
            objpoints.append(objp)

            if save_images_flag:
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

                if not os.path.exists("../output_images/chessboard_images"):
                    os.mkdir("../output_images/chessboard_images")
                # get filename of result image
                (_, tail) = os.path.split(file_name)
                (root, ext) = os.path.splitext(tail)
                result_filename = os.path.join(
                    "../output_images/chessboard_images", root + "-corners"
                    + ext)

                # save the result image
                mpimg.imsave(result_filename, img)

    # calibrate camera: calculate camera matrix mtx
    # and distortion coefficients dist
    # (takes shape from last image in loop)
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints,
                                             gray.shape[::-1], None, None)

    return mtx, dist
