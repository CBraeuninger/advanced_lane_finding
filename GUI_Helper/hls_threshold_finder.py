'''
Created on 1 oct. 2019

@author: cbraeuninger
'''
import cv2
from HLS_select import hls_select

class HLSThresholdFinder:    
    def __init__(self, image, thr_min=120, thr_max=255):
        self.original = image
        self._thr_min = thr_min
        self._thr_max = thr_max       

        def onchange_thr_min(pos):
            self._thr_min = pos
            self._render()

        def onchange_thr_max(pos):
            self._thr_max = pos
            self._render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('min. threshold', 'edges', self._thr_min, 255, onchange_thr_min)
        cv2.createTrackbar('max. threshold', 'edges', self._thr_max, 255, onchange_thr_max)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('original')

    def _render(self):
        
        self._edge_img = hls_select(self.original, (self._thr_min, self._thr_max))
        cv2.imshow('original', self.original)
        cv2.imshow('edges', self._edge_img)
        
    def thr_min(self):
        return self._thr_min
    
    def thr_max(self):
        return self._thr_max
