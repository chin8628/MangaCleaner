import logging
import math
import numpy as np
from tqdm import tqdm
import cv2
import logging
import math
import numpy as np


def get_letters(swt: np.ndarray, connected_components: dict):
    heights, widths, topleft_pts, swt_values = [], [], [], []

    for label, layer in connected_components.items():
        nz_y, nz_x = np.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        if height <= 10 or height >= 300:
            continue

        diameter = math.sqrt(width * width + height * height)
        median_swt = np.median(swt[(nz_y, nz_x)])

        if diameter / median_swt > 10 or diameter / median_swt < 1:
            continue

        heights.append(height)
        topleft_pts.append(np.asarray((north, west)))
        widths.append(width)
        swt_values.append(median_swt)

    logging.getLogger(__name__).info('Finished.')

    return heights, widths, topleft_pts, swt_values
