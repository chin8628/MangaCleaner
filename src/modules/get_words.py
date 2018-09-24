import logging
from typing import List, Tuple
import math

import scipy.spatial
import numpy as np


def get_words(swts: List[int], heights: List[int], widths: List[int], topleft_pts: Tuple[int, int]):
    swts_array = np.asarray([[math.log(i, 3)] for i in swts])
    heights_array = np.asarray([[math.log(i, 3)] for i in heights])
    topleft_pts_array = np.array(topleft_pts)

    if len(swts_array) == 0 or len(heights_array) == 0 or len(topleft_pts_array) == 0:
        return []

    swt_tree = scipy.spatial.KDTree(np.asarray(swts_array))
    stp = swt_tree.query_pairs(1)

    height_tree = scipy.spatial.KDTree(np.asarray(heights_array))
    htp = height_tree.query_pairs(1)

    isect = htp.intersection(stp)

    if len(isect) == 0:
        return []

    chains = []
    for idx1, idx2 in isect:
        widest = max(widths[idx1], widths[idx2])
        distance = np.linalg.norm(topleft_pts_array[idx1] - topleft_pts_array[idx2])
        if distance >= widest * 3:
            continue

        added = False
        for chain in chains:
            if idx1 not in chain and idx2 in chain:
                chain.append(idx1)
                added = True
            elif idx1 in chain and idx2 not in chain:
                chain.append(idx2)
                added = True

        if not added:
            chains.append([idx1, idx2])

    words = []
    for chain in chains:
        word_swts = []
        east_word, west_word, south_word, north_word = 0, np.inf, 0, np.inf

        for idx in chain:
            north, west = topleft_pts[idx]
            south, east = north + heights[idx], west + widths[idx]

            east_word = max(east_word, east)
            west_word = min(west_word, west)
            south_word = max(south_word, south)
            north_word = min(north_word, north)

            word_swts.append(swts[idx])

        width, height = east_word - west_word, south_word - north_word

        words.append({
            'swt': np.median(word_swts),
            'height': height,
            'width': width,
            'topleft_pt': (north_word, west_word)
        })

    logging.getLogger(__name__).info('Finished.')

    return words
