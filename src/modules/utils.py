import itertools
import logging
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def histogram_calculate_parallel(src: np.ndarray, words: Dict, hist_send_back: List) -> None:
    height, width = src.shape
    flatten_img = list(itertools.chain.from_iterable(src))

    for word in tqdm(words):
        y_start, y_stop = word['topleft_pt'][0], word['topleft_pt'][0] + word['height']
        x_start, x_stop = word['topleft_pt'][1], word['topleft_pt'][1] + word['width']
        selected = map(lambda y: flatten_img[(y * width) + x_start:(y * width) + x_stop], range(y_start, y_stop))

        hist = np.zeros(256).astype(np.uint8)
        for i in selected:
            hist[i] += 1

        hist_send_back.append(list(hist))


def histogram_calculate(src: np.ndarray) -> List:
    flatten_img = list(itertools.chain.from_iterable(src))

    hist = np.zeros(256).astype(np.uint8)
    for i in tqdm(flatten_img):
        hist[i] += 1

    return hist.copy()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
