import math
import numpy as np

def find_letters(swt, shapes):
    swts = []
    heights = []
    widths = []
    topleft_pts = []
    images = []

    for label, layer in shapes.items():
        (nz_y, nz_x) = np.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        if height < 10 or height > 300:
            continue

        diameter = math.sqrt(width * width + height * height)
        median_swt = np.median(swt[(nz_y, nz_x)])
        if diameter / median_swt > 10:
            continue

        # we use log_base_2 so we can do linear distance comparison later using k-d tree
        # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
        # Assumption: we've eliminated anything with median_swt == 1
        swts.append([math.log(median_swt, 2)])
        heights.append([math.log(height, 2)])
        topleft_pts.append(np.asarray([north, west]))
        widths.append(width)
        images.append(layer)

    print("len(swts):", len(swts))

    return swts, heights, widths, topleft_pts, images