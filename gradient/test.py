import matplotlib.image as mpimg
import glob
from gradient import combine_gradient
from visualization import save_result_image

# import images
# read in all images with names with pattern *.jpg
images = glob.glob('../test_images/*.jpg')

# loop over images and undistort them
for file_name in images:

    img = mpimg.imread(file_name)
    combined = combine_gradient(img)
    # save images
    save_result_image(combined, "../output_images/gradient_images",
                      file_name, "-gradient", True)
