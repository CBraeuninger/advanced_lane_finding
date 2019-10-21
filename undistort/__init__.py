import cv2


def undistort_image(img, mtx, dist):
    '''
    Undistorts image given a camera matrix and distortion coefficients
    '''

    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    return undist_img
