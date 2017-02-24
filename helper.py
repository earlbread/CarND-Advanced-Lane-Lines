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
    """Find image points and object points from given images.

    :param images: A list of opencv image array.
    :param nx: The number of corners in row.
    :param ny: The number of corners in column.

    :return: A tuple of object points and image points
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

    return objpoints, imgpoints


def undistort_image(image, objpoints, imgpoints):
    """Undistort image.

    :param image: Image to undistort.
    :param objpoints: 3D Object points in real world space.
    :param imgpoints: 2D Object points in image plane.

    :return: A undistorted image.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       image.shape[0:2],
                                                       None, None)

    undist = cv2.undistort(image, mtx, dist, None, mtx)

    return undist
