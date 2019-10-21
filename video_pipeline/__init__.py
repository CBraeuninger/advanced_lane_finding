from globals.fits import LineFit
from moviepy.editor import VideoFileClip
import numpy as np
from undistort import undistort_image
from perspective_transform import do_perspective_transform, warp_image
from lane_poly_fit import find_lanes, color_lane_pixels
from final_image import final_image
from binary_image import create_binary_image


def process_image(img):
    """
    Pipeline for processing each frame of the video
    """

    # ------------------------- define camera matrix and distortion coefficient
    mtx = np.array([[1.15777818e+03, 0.00000000e+00, 6.67113857e+02],
                    [0.00000000e+00, 1.15282217e+03, 3.86124583e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist_coeff = np.array([[-0.24688507, -0.02373155, -0.00109831,  0.00035107,
                            -0.00259868]])

    # --------------------------------------------------------- undistort image
    undist = undistort_image(img, mtx, dist_coeff)

    # --------------------------------------------------- generate binary image
    hls = create_binary_image(undist)

    # ------------------------------------- transform to bird's eye perspective
    warped, src, dst = do_perspective_transform(hls, line_fit)

    # ---------------------------------------------------------- fit polynomial
    leftx, lefty, rightx, righty, l_left_seg, l_right_seg,\
        leftx_base, rightx_base = find_lanes(warped)

    # ------------------------------ color detected lane pixels in warped image
    lane_pix = color_lane_pixels(np.zeros_like(warped), leftx, lefty, rightx,
                                 righty)

    # ------------------------------------- unwarp image of colored lane pixels
    lane_pix_unwarped = warp_image(lane_pix, dst, src)

    # ------superpose image of unwarped lane pixels image on original image and
    # -----------------------------add curvature and lane departure information
    res_img = final_image(img, lane_pix_unwarped, img.shape[0], leftx, lefty,
                          rightx, righty, l_left_seg, l_right_seg, src,
                          leftx_base, rightx_base)

    return res_img


# get video file name
video_name = 'project_video.mp4'

# read in video
clip1 = VideoFileClip("../"+video_name)

# get size of the frames in the video
vid_width, vid_height = clip1.size

line_fit = LineFit(vid_width, vid_height)

# process the frames with the lane detection pipeline
processed_clip = clip1.fl_image(process_image)

# write video to file
processed_clip.write_videofile("../output_videos/"+video_name, audio=False)
