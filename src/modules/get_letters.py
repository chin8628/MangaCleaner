import logging
import math

import numpy as np


def get_letters(swt: np.ndarray, connected_components: dict) -> dict:
    swt_values, heights, widths, topleft_pts, letter_images = [], [], [], [], []

    for label, layer in connected_components.items():
        (nz_y, nz_x) = np.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        # Letter shouldn't has border margin less than 5px
        margin = 5
        if north < margin or west < margin or swt.shape[1] - east < margin or swt.shape[0] - south < margin:
            continue

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

    logging.getLogger(__name__).info('Finished.')

    return swt_values, heights, widths, topleft_pts, letter_images
