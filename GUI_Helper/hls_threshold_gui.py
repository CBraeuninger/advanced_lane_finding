'''
Created on 1 oct. 2019

@author: cbraeuninger
'''
import cv2
from GUI_Helper.hls_threshold_finder import HLSThresholdFinder

filename = input('Give path to file:')

img = cv2.imread(filename)

hls_threshold_finder = HLSThresholdFinder(img)

print("Gradient thresholds:")
print("Min. threshold: %f" % hls_threshold_finder.thr_max())
print("Max. threshold: %f" % hls_threshold_finder.thr_min())
