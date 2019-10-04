import cv2
from GUI_Helper.gradient_finder import GradientParameterFinder

    
filename = input('Give path to file:')

img = cv2.imread(filename)

gradient_finder = GradientParameterFinder(img)

print("Gradient thresholds:")
print("Min. threshold x-gradient: %f" % gradient_finder.x_tr_min())
print("Max. threshold x-gradient: %f" % gradient_finder.x_tr_max())
print("Min. threshold y-gradient: %f" % gradient_finder.y_tr_min())
print("Max. threshold y-gradient: %f" % gradient_finder.y_tr_max())
print("Min. threshold gradient magnitude: %f" % gradient_finder.mag_tr_min())
print("Max. threshold gradient magnitude: %f" % gradient_finder.mag_tr_max())
print("Min. threshold gradient direction: %f" % gradient_finder.dir_tr_min())
print("Max. threshold gradient direction: %f" % gradient_finder.dir_tr_max())

