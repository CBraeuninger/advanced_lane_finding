import glob
import cv2
from perspective_transform import do_perspective_transform
from visualization import save_result_image
from globals.fits import LineFit

# import images
# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    img = cv2.imread(file_name)

    # initialize line_fit
    line_fit = LineFit(img.shape[1], img.shape[0])

    warped, src, dst = do_perspective_transform(img, line_fit)
    # save images
    save_result_image(warped, "../output_images/warped",
                      file_name, "-warped", True)
