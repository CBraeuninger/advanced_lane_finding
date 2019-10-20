'''
Created on 13 oct. 2019

@author: cbraeuninger
'''
from VideoPipeline.Fit import LineFit
from moviepy.editor import VideoFileClip
import numpy as np
from Undistort_Image import undistort_image
from HLS_select import saturation_select
from PerspectiveTransform import doPerspectiveTransform, warpImage
from LanePolyFit import findLanes, colorLanePixels
from FinalImage import finalImage
from BinaryImage import createBinaryImage


def processImage(img):
    
    #--------------------------- define camera matrix and distortion coefficient
    mtx = np.array([[1.15777818e+03, 0.00000000e+00, 6.67113857e+02], \
                    [0.00000000e+00, 1.15282217e+03, 3.86124583e+02], \
                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    dist_coeff = np.array([[-0.24688507, -0.02373155, -0.00109831,  0.00035107, -0.00259868]])
    
    #----------------------------------------------------------- undistort image
    undist = undistort_image(img, mtx, dist_coeff)    
    
    #--- convert to hls color space and apply threshold to generate binary image
    hls = createBinaryImage(undist)
    
    #-------------------------------------- transform to bird's eye perspective
    warped, src, dst = doPerspectiveTransform(hls, lineFit)
     
    #------------------------------------------------------------ fit polynomial
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg, leftx_base, rightx_base = findLanes(warped)
     
    #-------------------------------- color detected lane pixels in warped image
    lanePix = colorLanePixels(np.zeros_like(warped), leftx, lefty, rightx, righty)
     
    #-------------------------------------------------------------- unwarp image
    lanePixUnwarped = warpImage(lanePix, dst, src)
     
    # superpose image of unwarped lane pixels image on original image and add curvature
    res_img = finalImage(img, lanePixUnwarped, img.shape[0], leftx, lefty, rightx, righty, l_left_seg, l_right_seg, src, leftx_base, rightx_base)
    
    return res_img

#get video file name
video_name = 'project_video.mp4'

#read in video
clip1 = VideoFileClip("../"+video_name)

vid_width, vid_height = clip1.size

lineFit = LineFit(vid_width, vid_height)

#process the frames with the lane detection pipeline
processed_clip = clip1.fl_image(processImage) 

#write video to file
processed_clip.write_videofile("../output_videos/"+video_name, audio=False)