import cv2
import numpy as np


def abs_sobel_thresh(image, orient='x', ksize=3, thresh=(0, 255)):
    """Generate a directional gradient binary image.

    :param image: An image to generate binary image.
    :param orient: Gradient direction, 'x' or 'y'.
    :param ksize: Kernel size to apply sobel.
    :param thresh: Low and high gradient threshold.

    :return: Directional graident binary image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, ksize=3, thresh=(0, 255)):
    """Generate the magnitude of the gradient.

    :param image: An image to generate binary image.
    :param ksize: Kernel size to apply sobel.
    :param thresh: Low and high gradient threshold.

    :return: The maginitude of the gradient.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255*abs_sobelxy / np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mag_binary


def dir_thresh(image, ksize=3, thresh=(0, np.pi/2)):
    """Generate the direction of the gradient.

    :param image: An image to generate binary image.
    :param ksize: Kernel size to apply sobel.
    :param thresh: Low and high gradient threshold.

    :return: The direction of the gradient.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def ch_thresh(ch, thresh=(0, 255)):
    """Generate a channel binary image.

    :param ch: A color channel to generate binary image.
    :param thresh: Low and high threshold.

    :return: A channel binary image.
    """
    ch_binary = np.zeros_like(ch)
    ch_binary[(ch > thresh[0]) & (ch <= thresh[1])] = 1
    return ch_binary


def color_combine(image):
    """Generate a binary image using color thresholding.

    :param image: An image to generate binary image.

    :return: A binary image.
    """

    thresh = (170, 255)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    combined = np.zeros_like(s)
    combined[(s > thresh[0]) & (s <= thresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    h_binary = ch_thresh(h, (100, 255))
    v_binary = ch_thresh(v, (220, 255))
    s_binary = ch_thresh(s, (150, 255))

    r = image[:, :, 0]
    r_binary = ch_thresh(r, (205, 255))

    combined = np.zeros_like(s_binary)
    combined[((s_binary == 1) | (v_binary == 1) | (r_binary == 1)) & (h_binary == 0)] = 1

    return combined


def grad_combine(image):
    """Generate a binary image using gradient thresholding.

    :param image: An image to generate binary image.

    :return: A binary image.
    """
    ksize = 9
    gradx = abs_sobel_thresh(image, orient='x', ksize=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(image, orient='y', ksize=ksize, thresh=(20, 255))

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1))] = 1

    return combined


def thresh_combine(image):
    """Generate a combined binary image of gradient and color.

    :param image: An image to generate binary image.

    :return: A binary image.
    """

    color = color_combine(image)
    grad = grad_combine(image)

    combined = np.zeros_like(grad)
    combined[(color == 1) | (grad == 1)] = 1

    return combined
