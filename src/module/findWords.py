import scipy.sparse
import scipy.spatial
import numpy as np

def find_words(swts, heights, widths, topleft_pts, images):
    swt_tree = scipy.spatial.KDTree(np.asarray(swts))
    stp = swt_tree.query_pairs(1)

    height_tree = scipy.spatial.KDTree(np.asarray(heights))
    htp = height_tree.query_pairs(1)

    isect = htp.intersection(stp)

    chains = []
    pairs = []
    pair_angles = []
    for pair in isect:
        left = pair[0]
        right = pair[1]
        widest = max(widths[left], widths[right])
        distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
        if distance < widest * 1.5:
            delta_yx = topleft_pts[left] - topleft_pts[right]
            angle = np.arctan2(delta_yx[0], delta_yx[1])
            if angle < 0:
                angle += np.pi

            pairs.append(pair)
            pair_angles.append(np.asarray([angle]))

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
            chains.append(set([left_a, right_a]))

        added = False
        for chain in chains:
            if left_b in chain:
                chain.add(right_b)
                added = True
            elif right_b in chain:
                chain.add(left_b)
                added = True

        if not added:
            chains.append(set([left_b, right_b]))

    word_images = []
    for chain in set([tuple(i) for i in chains]):
        word = []
        for idx in chain:
            word.append(images[idx])
        word_images.append(word)

    return word_images
