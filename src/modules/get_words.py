import logging
import scipy.sparse
import scipy.spatial
import numpy as np


def get_words(swts, heights, widths, topleft_pts):
    swts_array = np.asarray([[math.log(i, 2)] for i in swts])
    heights_array = np.asarray([[math.log(i, 2)] for i in heights])

    swt_tree = scipy.spatial.KDTree(swts_array)
    stp = swt_tree.query_pairs(1)

    height_tree = scipy.spatial.KDTree(heights_array)
    htp = set(height_tree.query_pairs(1))

    isect = htp.intersection(stp)

    chains = []
    pairs = []
    pair_angles = []
    for pair in isect:
        left = pair[0]
        right = pair[1]
        widest = max(widths[left], widths[right])
        distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
        if distance < widest * 3:
            delta_yx = topleft_pts[left] - topleft_pts[right]
            angle = np.arctan2(delta_yx[0], delta_yx[1])
            if angle < 0:
                angle += np.pi

            pairs.append(pair)
            pair_angles.append(np.asarray([angle]))

    if len(pair_angles) == 0:
        return []

    angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
    atp = angle_tree.query_pairs(np.pi / 12)

    for pair_idx in atp:
        pair_a = pairs[pair_idx[0]]
        pair_b = pairs[pair_idx[1]]
        left_a = pair_a[0]
        right_a = pair_a[1]
        left_b = pair_b[0]
        right_b = pair_b[1]

        added = False
        for chain in chains:
            if left_a in chain:
                chain.add(right_a)
                added = True
            elif right_a in chain:
                chain.add(left_a)
                added = True

        if not added:
            chains.append({left_a, right_a})

        added = False
        for chain in chains:
            if left_b in chain:
                chain.add(right_b)
                added = True
            elif right_b in chain:
                chain.add(left_b)
                added = True

        if not added:
            chains.append({left_b, right_b})

    words = []

    for chain in set([tuple(i) for i in chains]):
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
