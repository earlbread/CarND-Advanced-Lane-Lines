import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def process_image(image, mtx, dist):
    """Process image to identify the lane boundaries.

    :param image: Image to processs.
    :param mtx: Camera matrix.
    :param dist: Distortion coefficients.

    :return: A processed image.
    """
    image = undistort_image(image, mtx, dist)
    return image


def process_video(video_filename, mtx, dist, prefix="precessed_"):
    """Process video to identify the lane boundaries.

    :param video: Video file name to process.
    :param mtx: Camera matrix.
    :param dist: Distortino coefficients.
    """
    def process_fl(image):
        return process_image(image, mtx, dist)
    clip = VideoFileClip(video_filename)
    new_clip = clip.fl_image(process_fl)
    new_filename = prefix + video_filename
    new_clip.write_videofile(new_filename)
