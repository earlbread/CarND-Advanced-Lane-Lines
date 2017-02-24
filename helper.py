import cv2
import numpy as np


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
