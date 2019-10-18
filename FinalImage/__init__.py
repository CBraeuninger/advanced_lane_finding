from Curvature import realLaneCurvature, distanceFromLane, getConversion
import cv2
import numpy as np
from VideoPipeline import config

def addLaneLines(img, lane_line_img):
    
    ret_img = img.copy()
    ret_img[(lane_line_img[:,:,0] > 0)] = (255,0,0)
    ret_img[(lane_line_img[:,:,2] > 0)] = (0,0,255)
    
    return ret_img

def addInfo(img, yeval, leftx, lefty, rightx, righty, l_left_seg, l_right_seg, leftx_base, rightx_base, src):
    
    ym_per_pix, xm_per_pix = getConversion(img.shape[0], leftx_base, rightx_base)
    
    config.curvature.append_array(realLaneCurvature(yeval, leftx, lefty, rightx, righty, l_left_seg, l_right_seg, ym_per_pix, xm_per_pix), 100)
    
    avg_curvature = config.curvature.average()
    
    config.offset.append_array(distanceFromLane(img, src), 100)
    
    avg_offset = config.offset.average()
    
    if avg_offset>0:
        side = "left"
    else:
        side = "right"
    
    img = cv2.putText(img, "Curvature radius in m: {0:.2f} m".format(avg_curvature), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2,\
                       (255,255,255), bottomLeftOrigin=False)
    img = cv2.putText(img, "Vehicle is {0:.2f} m ".format(abs(avg_offset)) + side + " of center.", (50,100), cv2.FONT_HERSHEY_PLAIN, 2,\
                      (255,255,255), bottomLeftOrigin=False)
    
    return img

def addTrapezoid(img, src):    
    
    points = np.array([[(src[0][0],src[0][1]), (src[1][0],src[1][1]), (src[2][0], src[2][1]), (src[3][0],src[3][1])]], dtype=np.int32)
    
    img = cv2.fillPoly(img, points, (0,255,0))
    
    return img

def finalImage(img, lane_line_img, yeval, leftx, lefty, rightx, righty, l_left_seg, l_right_seg, src, leftx_base, rightx_base):
    
    img = addLaneLines(img, lane_line_img)
    img = addInfo(img, yeval, leftx, lefty, rightx, righty, l_left_seg, l_right_seg, leftx_base, rightx_base, src)
    trapImg = addTrapezoid(np.zeros_like(img), src)
    
    resImg = cv2.addWeighted(img, 1.0, trapImg, 0.5, 0.0)
    
    return resImg