'''
Created on 14 oct. 2019

@author: cbraeuninger
'''
import numpy as np

class Fit:

    def __init__(self):    
        self._l_fit = np.array([0,0,0])
        self._r_fit = np.array([0,0,0])
    
    def get_l_fit(self):
        return self._l_fit
    
    def get_r_fit(self):
        return self._r_fit
    
    def set_l_fit(self, new_l_fit):
        self._l_fit = new_l_fit
        
    def set_r_fit(self, new_r_fit):
        self._r_fit = new_r_fit
        
class FitReal:

    def __init__(self):    
        self._l_fit = np.array([0,0,0])
        self._r_fit = np.array([0,0,0])
    
    def get_l_fit(self):
        return self._l_fit
    
    def get_r_fit(self):
        return self._r_fit
    
    def set_l_fit(self, new_l_fit):
        self._l_fit = new_l_fit
        
    def set_r_fit(self, new_r_fit):
        self._r_fit = new_r_fit