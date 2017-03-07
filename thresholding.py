import cv2
import numpy as np


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Generate a directional gradient binary image.

    :param image: An image to generate binary image.
    :param orient: Gradient direction, 'x' or 'y'.
    :param sobel_kernel: Kernel size to apply sobel.
    :param thresh: Low and high gradient threshold.

    :return: Directional graident binary image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """Generate the magnitude of the gradient.

    :param image: An image to generate binary image.
    :param sobel_kernel: Kernel size to apply sobel.
    :param thresh: Low and high gradient threshold.

    :return: The maginitude of the gradient.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255*abs_sobelxy / np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """Generate the direction of the gradient.

    :param image: An image to generate binary image.
    :param sobel_kernel: Kernel size to apply sobel.
    :param thresh: Low and high gradient threshold.

    :return: The direction of the gradient.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def s_thresh(image, thresh=(0, 255)):
    """Generate S channel binary image in HLS.

    :param image: An image to generate binary image.
    :param thresh: Low and high threshold.

    :return: S channel binary image in HLS.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    s_binary = np.zeros_like(s)
    s_binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    return s_binary
