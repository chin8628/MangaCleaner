import logging
from typing import Dict, List

import numpy as np


def histogram_calculate_parallel(src: np.ndarray, words: Dict, hist_send_back: List) -> None:
    height, width, channel = src.shape
    flatten_img = src.ravel()

    for word in words:
        y_start, y_stop = word['topleft_pt'][0], word['topleft_pt'][0] + word['height']
        x_start, x_stop = word['topleft_pt'][1], word['topleft_pt'][1] + word['width']
        selected = []
        for y in range(y_start, y_stop):
            selected += list(flatten_img[(y * width) + x_start:(y * width) + x_stop])

        hist = [selected.count(i) for i in range(0, 256)]
        sum_hist = sum(hist)

        hist_send_back.append([round(i / sum_hist, 4) for i in hist])
