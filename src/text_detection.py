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
    heights, widths, topleft_pts, swt_values = get_letters(swt, connected_components)
    words = get_words(swt_values, heights, widths, topleft_pts)

    for word in words:
        yield {
            'x': word['topleft_pt'][1],
            'y': word['topleft_pt'][0],
            'width': word['width'],
            'height': word['height']
        }
