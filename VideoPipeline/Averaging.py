'''
Created on 18 oct. 2019

@author: cbraeuninger
'''
from numpy import average

class Average:
    
    def __init__(self):
        self._array = []
        
    def get_array(self):
        return self._array
    
    def append_array(self, new_value, n_frames):
        
        if len(self._array) >= n_frames:
            self._array.pop(0)
            
        self._array.append(new_value)
        
    def average(self):
        return average(self._array)
    

        