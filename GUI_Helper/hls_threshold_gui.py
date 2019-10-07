'''
Created on 1 oct. 2019

@author: cbraeuninger
'''
import cv2
import glob
import numpy as np
from GUI_Helper.hls_threshold_finder import HLSThresholdFinder

filenames = glob.glob('../test_images/*.jpg')

min_thr_array = []
max_thr_array = []

for filename in filenames:
    
    img = cv2.imread(filename)
    
    hls_threshold_finder = HLSThresholdFinder(img)
    
    print("Gradient thresholds:")
    print("Min. threshold: %f" % hls_threshold_finder.thr_max())
    min_thr_array.append(hls_threshold_finder.thr_max())
    print("Max. threshold: %f" % hls_threshold_finder.thr_min())
    max_thr_array.append(hls_threshold_finder.thr_min())
    
mean_min_thr = np.average(min_thr_array)
mean_max_thr = np.average(max_thr_array)
print("Mean values for thresholds: max: '{0}', min: '{1}'".format(mean_min_thr, mean_max_thr))

file_obj = open("hls_threshold.txt", 'w')
file_obj.write("Mean values for thresholds: max: '{0}', min: '{1}'".format(mean_min_thr, mean_max_thr))
file_obj.close()