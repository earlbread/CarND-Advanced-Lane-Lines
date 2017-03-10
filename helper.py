import argparse
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

import calibration as calib
import thresholding
import perspective
import laneline


def process_image(image, mtx, dist):
    """Process image to identify the lane boundaries.

    :param image: Image to processs.
    :param mtx: Camera matrix.
    :param dist: Distortion coefficients.

    :return: A processed image.
    """
    image = calib.undistort_image(image, mtx, dist)
    binary = thresholding.thresh_combine(image)
    binary_warped = perspective.perspective_transform(binary)
    detected = laneline.sliding_window(binary_warped)

    return detected


def process_video(video_filename, mtx, dist, prefix="processed_"):
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


if __name__ == '__main__':
    mtx, dist = calib.get_calibration_info('./camera_cal', 9, 6)
    parser = argparse.ArgumentParser(description='Create processed video.')
    parser.add_argument(
        'video_filename',
        type=str,
        default='',
        help='Path to video file.'
    )
    args = parser.parse_args()

    video_filename = args.video_filename
    process_video(video_filename, mtx, dist)
