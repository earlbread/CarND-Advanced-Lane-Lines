import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def get_images(files):
    """Read images using OpenCV from given a list of files.

    :param files: A list of image file name.

    :return: A list of opencv image array.
    """
    images = []

    for f in files:
        image = cv2.imread(f)
        images.append(image)

    return images


def get_calibration_info(images, nx, ny):
    """Find camera matrix and distortion coefficients to undistort image.

    :param images: A list of opencv image array.
    :param nx: The number of corners in row.
    :param ny: The number of corners in column.

    :return: A tuple of camera matrix and distortion coefficients.
    """
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    # ret, rvecs and tvecs are not used.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None, None)

    return mtx, dist


def undistort_image(image, mtx, dist):
    """Undistort image.

    :param image: Image to undistort.
    :param mtx: Camera matrix.
    :param dist: Distortion coefficients.

    :return: A undistorted image.
    """
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    return undist


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
