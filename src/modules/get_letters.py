import logging
import math

import numpy as np
from tqdm import tqdm


def get_letters(swt: np.ndarray, connected_components: dict):
    swt_values, heights, widths, diameters, topleft_pts = [], [], [], [], []

    cnt = 0

    for label, layer in tqdm(connected_components.items()):
        # if cnt >= 100:
        #     break

        nz_y, nz_x = layer.nonzero()
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        diameter = math.sqrt(width * width + height * height)
        median_swt = np.median(swt[(nz_y, nz_x)])

        if (height <= 5 or height >= 300) and (width <= 5 or width >= 300):
            continue

        if diameter / median_swt <= 1 or diameter / median_swt > 20:
            continue

        swt_values.append(median_swt)
        heights.append(height)
        topleft_pts.append((north, west))
        widths.append(width)
        diameters.append(diameter)

        cnt += 1

    logging.getLogger(__name__).info('Finished.')

    return swt_values, heights, widths, diameters, topleft_pts
