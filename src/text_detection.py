from typing import List

import cv2
import numpy as np
import math

# Modules
from modules.get_connected_components import get_connected_components
from modules.get_letters import get_letters
from modules.get_swt import get_swt


def text_detection(src: np.ndarray, expected_height: int = 1200) -> List[dict]:
    if src.shape[0] > expected_height:
        print('Image data is too height. (It\' exceeds {}px)'.format(expected_height))
        quit()

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height, width = src.shape[:2]

    high_thresh, thresh_im = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    edges = cv2.Canny(src_gray, low_thresh, high_thresh)

    invert_gray_image = cv2.bitwise_not(src_gray)
    sobel_x = cv2.Sobel(invert_gray_image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(invert_gray_image, cv2.CV_64F, 0, 1)

    magnitude, direction = cv2.cartToPolar(sobel_x, sobel_y)

    swt = get_swt(edges, sobel_x, sobel_y, direction, magnitude, height, width)
    connected_components, label_map = get_connected_components(swt)
    topleft_pts, heights, widths = get_letters(swt, connected_components)

    mask = np.zeros(src_gray.shape, np.uint8)
    for index in range(0, len(topleft_pts)):
        x1, y1 = topleft_pts[index][1], topleft_pts[index][0]
        x2, y2 = topleft_pts[index][1] + widths[index], topleft_pts[index][0] + heights[index]
        mask[y1:y2, x1:x2] = 1

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x1, y1, w, h = cv2.boundingRect(cnt)
        x2, y2 = x1 + w, y1 + h

        if w < 5 or h < 5 or w > 60 or h > 60:
            continue

        margin = 3
        diff = (max(h, w) - min(h, w)) / 2
        if h < w:
            y1, y2 = int(y1 - diff), int(y2 + diff)
        elif w < h:
            x1, x2 = int(x1 - diff), int(x2 + diff)

        x1, y1 = x1-margin, y1-margin
        x2, y2 = x2+margin, y2+margin
        w, h = x2 - x1, y2 - y1

        yield {'x': x1, 'y': y1, 'width': w, 'height': h}
