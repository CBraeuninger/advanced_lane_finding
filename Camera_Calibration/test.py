'''
Created on 25 sept. 2019

@author: cbraeuninger
'''
from Camera_Calibration import calibrate_camera

mtx, dist = calibrate_camera(False)

print("camera matrix:")
print(mtx)
print("distortion coefficients:")
print(dist)