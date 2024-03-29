import logging
import math
import numpy as np
from tqdm import tqdm
import cv2
import logging
import math
import numpy as np


def get_letters(swt: np.ndarray, connected_components: dict):
    heights, widths, topleft_pts = [], [], []

    for label, layer in tqdm(connected_components.items()):
        nz_y, nz_x = np.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        if height <= 1 or height >= 50:
            continue

        if width <= 1 or width >= 50:
            continue

        diameter = math.sqrt(width * width + height * height)
        median_swt = np.median(swt[(nz_y, nz_x)])

        if diameter / median_swt >= 15 or diameter / median_swt <= 1:
            continue

        if median_swt > 8:
            continue

        heights.append(height)
        topleft_pts.append((north, west))
        widths.append(width)

    logging.getLogger(__name__).info('Finished.')

    return topleft_pts, heights, widths
