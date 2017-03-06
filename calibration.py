import cv2
import glob
import numpy as np


def get_calibration_info(image_path, nx, ny):
    """Find camera matrix and distortion coefficients to undistort image.

    :param image_path: The path of chessboard images.
    :param nx: The number of corners in row.
    :param ny: The number of corners in column.

    :return: A tuple of camera matrix and distortion coefficients.
    """
    images = []
    for f in glob.glob(image_path + '/*.jpg'):
        image = cv2.imread(f)
        images.append(image)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
