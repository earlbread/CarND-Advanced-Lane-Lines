import cv2
import numpy as np


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

    src = np.float32([[(167, 720),
                     (585, 453),
                     (698, 453),
                     (1160, 720)]])

    dst = np.float32([[(320, 720),
                       (320, 0),
                       (960, 0),
                       (960, 720)]])

    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (imshape[1], imshape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
