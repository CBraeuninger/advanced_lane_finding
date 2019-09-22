'''
Created on 16 sept. 2019

@author: cbraeuninger
'''
from camera_calibration import calibrate_camera

mtx, dist = calibrate_camera(False)

print("camera matrix:")
print(mtx)
print("distortion coefficients:")
print(dist)
    
