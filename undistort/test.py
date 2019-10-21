import glob
import matplotlib.image as mpimg
from camera_calibration import calibrate_camera
from undistort import undistort_image
from visualization import save_result_image

# get calibration matrix
mtx, dst = calibrate_camera()

# read in all images with names with pattern calibration*.jpg
images = glob.glob('../camera_cal/calibration*.jpg')

# loop over images and undistort them
for file_name in images:
    # read in image
    img = mpimg.imread(file_name)
    # undistort image
    undist_img = undistort_image(img, mtx, dst)
    # save resulting image
    save_result_image(undist_img, "../output_images/undistorted_cal_images", file_name, "-undist")
