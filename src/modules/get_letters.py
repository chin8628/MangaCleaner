import logging
import math

import numpy as np
from tqdm import tqdm

import cv2


def get_letters(swt: np.ndarray, connected_components: dict):
    swt_values, heights, widths, diameters, topleft_pts = [], [], [], [], []

    for label, layer in tqdm(connected_components.items()):
        nz_y, nz_x = np.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        if height <= 5 or height >= 300:
            continue

        diameter = math.sqrt(width * width + height * height)
        median_swt = np.median(swt[(nz_y, nz_x)])

        if diameter / median_swt >= 10:
            continue

        swt_values.append(median_swt)
        heights.append(height)
        topleft_pts.append((north, west))
        widths.append(width)
        diameters.append(diameter)

    logging.getLogger(__name__).info('Finished.')

    return swt_values, heights, widths, diameters, topleft_pts
