import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def region_of_interest(img, vertices):
    """Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    :param image: An image to apply an image mask.
    :param vertices: A numpy array of vertex to form the region.

    :return: A masked image.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def perspective_transform(image):
    """Applies perspective transform.

    :param image: An image to apply perspective transform.

    :return: A warped image.
    """
    imshape = image.shape

    x_left_bottom_src  = imshape[1] * 0.14
    x_left_top_src     = imshape[1] * 0.458
    x_right_top_src    = imshape[1] * 0.542
    x_right_bottom_src = imshape[1] * 0.88

    y_top_src          = imshape[0] * 0.63
    y_bottom_src       = imshape[0]

    src = np.float32([[(x_left_bottom_src, y_bottom_src),
                          (x_left_top_src, y_top_src),
                          (x_right_top_src, y_top_src),
                          (x_right_bottom_src, y_bottom_src)]])

    x_left_bottom_dst  = imshape[1] * 0.25
    x_left_top_dst     = x_left_bottom_dst
    x_right_top_dst    = imshape[1] * 0.75
    x_right_bottom_dst = x_right_top_dst

    y_top_dst          = 0
    y_bottom_dst       = imshape[0]

    dst = np.float32([[(x_left_bottom_dst, y_bottom_dst),
                          (x_left_top_dst, y_top_dst),
                          (x_right_top_dst, y_top_dst),
                          (x_right_bottom_dst, y_bottom_dst)]])

    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (imshape[1], imshape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


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
