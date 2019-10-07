import cv2
from GUI_Helper.gradient_finder import GradientParameterFinder
import glob
import numpy as np
    
filenames = glob.glob('../test_images/*.jpg')

x_tr_min_array = []
x_tr_max_array = []
y_tr_min_array = []
y_tr_max_array = []
mag_tr_min_array = []
mag_tr_max_array = []
dir_tr_min_array = []
dir_tr_max_array = []

for filename in filenames:
    
    img = cv2.imread(filename)
    
    gradient_finder = GradientParameterFinder(img)
    
    print("Gradient thresholds:")
    print("Min. threshold x-gradient: %f" % gradient_finder.x_tr_min())
    x_tr_min_array.append(gradient_finder.x_tr_min())
    print("Max. threshold x-gradient: %f" % gradient_finder.x_tr_max())
    x_tr_max_array.append(gradient_finder.x_tr_max())
    print("Min. threshold y-gradient: %f" % gradient_finder.y_tr_min())
    y_tr_min_array.append(gradient_finder.y_tr_max())
    print("Max. threshold y-gradient: %f" % gradient_finder.y_tr_max())
    y_tr_max_array.append(gradient_finder.y_tr_max())
    print("Min. threshold gradient magnitude: %f" % gradient_finder.mag_tr_min())
    mag_tr_min_array.append(gradient_finder.mag_tr_min())
    print("Max. threshold gradient magnitude: %f" % gradient_finder.mag_tr_max())
    mag_tr_max_array.append(gradient_finder.mag_tr_max())
    print("Min. threshold gradient direction: %f" % gradient_finder.dir_tr_min())
    dir_tr_min_array.append(gradient_finder.dir_tr_min())
    print("Max. threshold gradient direction: %f" % gradient_finder.dir_tr_max())
    dir_tr_max_array.append(gradient_finder.dir_tr_max())
    
mean_x_tr_min = np.average(x_tr_min_array)
mean_x_tr_max = np.average(x_tr_max_array)
mean_y_tr_min = np.average(y_tr_min_array)
mean_y_tr_max = np.average(y_tr_max_array)
mean_mag_tr_min = np.average(mag_tr_min_array)
mean_mag_tr_max = np.average(mag_tr_max_array)
mean_dir_tr_min = np.average(dir_tr_min_array)
mean_dir_tr_max = np.average(dir_tr_max_array)

print("Mean values for thresholds:")
print("Min. threshold x-gradient: '{0}'".format(mean_x_tr_min))    
print("Max. threshold x-gradient: '{0}'".format(mean_x_tr_max))
print("Min. threshold y-gradient: '{0}'".format(mean_y_tr_min))
print("Max. threshold y-gradient: '{0}'".format(mean_y_tr_max))
print("Min. threshold gradient magnitude: '{0}'".format(mean_mag_tr_min))
print("Max. threshold gradient magnitude: '{0}'".format(mean_mag_tr_max))
print("Min. threshold gradient direction: '{0}'".format(mean_dir_tr_min))
print("Max. threshold gradient direction: '{0}'".format(mean_dir_tr_max))

file_obj = open("gradient_thresholds.txt", 'w')
file_obj.write("Mean values for thresholds:\n")
file_obj.write("Min. threshold x-gradient: '{0}'\n".format(mean_x_tr_min))    
file_obj.write("Max. threshold x-gradient: '{0}'\n".format(mean_x_tr_max))
file_obj.write("Min. threshold y-gradient: '{0}'\n".format(mean_y_tr_min))
file_obj.write("Max. threshold y-gradient: '{0}'\n".format(mean_y_tr_max))
file_obj.write("Min. threshold gradient magnitude: '{0}'\n".format(mean_mag_tr_min))
file_obj.write("Max. threshold gradient magnitude: '{0}'\n".format(mean_mag_tr_max))
file_obj.write("Min. threshold gradient direction: '{0}'\n".format(mean_dir_tr_min))
file_obj.write("Max. threshold gradient direction: '{0}'\n".format(mean_dir_tr_max))

file_obj.close()