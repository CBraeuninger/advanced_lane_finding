import glob
import cv2
from perspective_transform import lane_line_mask
from visualization import save_result_image

# import images
# read in all binary images with names with pattern *.jpg
images = glob.glob('../output_images/hls/*.jpg')

# loop over images and undistort them
for file_name in images:

    # read in example image
    img = cv2.imread(file_name)

    # mask the image
    masked = lane_line_mask(img)

    # save images
    save_result_image(masked, "../output_images/Masked", file_name, "-masked",
                      True)
