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

    dst = np.float32([[(200, 720),
                       (200, 0),
                       (1080, 0),
                       (1080, 720)]])

    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (imshape[1], imshape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


def first_order_filter(prev_pts, next_pts):
    ALPHA  = 0.2
    next_pts = prev_pts * (1 - ALPHA) + next_pts * ALPHA
    return next_pts


def fill_laneline(image, binary_warped, ploty, left_fitx, right_fitx, prev_left, prev_right):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    src = np.float32([[(167, 720),
                     (585, 453),
                     (698, 453),
                     (1160, 720)]])

    dst = np.float32([[(200, 720),
                       (200, 0),
                       (1080, 0),
                       (1080, 720)]])

    Minv = cv2.getPerspectiveTransform(dst, src)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    if prev_left is not None:
        if np.absolute(np.average(prev_left - pts_left)) < 50:
            pts_left = first_order_filter(prev_left, pts_left)
        else:
            pts_left = prev_left

    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    if prev_right is not None:
        if np.absolute(np.average(prev_right - pts_right)) < 50:
            pts_right = first_order_filter(prev_right, pts_right)
        else:
            pts_right = prev_right

    if (prev_left is not None and
        prev_right is not None and
        not 200 < np.absolute(np.average(pts_left - pts_right)) < 600):
        pts_left = prev_left
        pts_right = prev_right

    prev_left = pts_left
    prev_right = pts_right

    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result, prev_left, prev_right
