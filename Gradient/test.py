'''
Created on 28 sept. 2019

@author: cbraeuninger
'''

import matplotlib.image as mpimg
import glob
from Gradient import combine_gradient
from VisualizationHelpers import saveResultImage

#import images
#read in all images with names with pattern *.jpg
images = glob.glob('../test_images/*.jpg')

# loop over images and undistort them
for file_name in images:
    
    img = mpimg.imread(file_name)
    combined = combine_gradient(img)
    #save images
    saveResultImage(combined, "../output_images/gradient_images", file_name, "-gradient", True)