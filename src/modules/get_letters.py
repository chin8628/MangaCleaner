import logging
import math

import numpy as np
from tqdm import tqdm


def get_letters(swt: np.ndarray, connected_components: dict):
    swt_values, heights, widths, diameters, topleft_pts, letter_images, hw_ratio = [], [], [], [], [], [], []

    for label, layer in tqdm(connected_components.items()):
        nz_y, nz_x = np.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        del nz_y
        del nz_x

        if (height <= 5 or height >= 300) and (width <= 5 or width >= 300):
            continue

        diameter = math.sqrt(width * width + height * height)
        median_swt = np.median(swt[(nz_y, nz_x)])
        if diameter / median_swt <= 1 or diameter / median_swt > 20:
            continue

        swt_values.append(median_swt)
        heights.append(height)
        topleft_pts.append((north, west))
        widths.append(width)
        letter_images.append(layer)
        diameters.append(diameter)
        hw_ratio.append(height / width)

    logging.getLogger(__name__).info('Finished.')

    return swt_values, heights, widths, diameters, topleft_pts, letter_images, hw_ratio
