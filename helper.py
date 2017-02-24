import cv2
import numpy as np


def get_images(files):
    """Read images using opencv from list of files
    """
    images = []

    for f in files:
        image = cv2.imread(f)
        images.append(image)

    return images


def get_calibration_info(images, nx, ny):
    """Return image points and object point to camera calibration
    """
    imgpoints = []  # 2D points in image plane
    objpoints = []  # 3D points in real world space

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    return imgpoints, objpoints
