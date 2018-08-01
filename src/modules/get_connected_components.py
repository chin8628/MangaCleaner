import logging

import numpy as np


def get_connected_components(swt: np.ndarray):
    class Label(object):
        def __init__(self, value):
            self.value = value
            self.parent = self
            self.rank = 0

    ld = {}

    def make_set(x):
        try:
            return ld[x]
        except KeyError:
            item = Label(x)
            ld[x] = item
            return item

    def find(item):
        if item.parent != item:
            item.parent = find(item.parent)

        return item.parent

    def union(x, y):
        """
        :param x:
        :param y:
        :return: root node of new union tree
        """
        x_root = find(x)
        y_root = find(y)
        if x_root == y_root:
            return x_root

        if x_root.rank < y_root.rank:
            x_root.parent = y_root
            return y_root
        elif x_root.rank > y_root.rank:
            y_root.parent = x_root
            return x_root
        else:
            y_root.parent = x_root
            x_root.rank += 1
            return x_root

    trees = {}
    label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
    next_label = 1

    swt_ratio_threshold = 3.0

    for y in range(swt.shape[0]):
        for x in range(swt.shape[1]):

            sw_point = swt[y, x]
            if 0 < sw_point < np.Infinity:
                neighbors = [(y, x - 1),  # west
                             (y - 1, x - 1),  # northwest
                             (y - 1, x),  # north
                             (y - 1, x + 1)]  # northeast
                connected_neighbors = None
                neighbor_values = []

                for neighbor in neighbors:
                    try:
                        sw_n = swt[neighbor]
                        label_n = label_map[neighbor]
                    except IndexError:
                        continue
                    if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                        neighbor_values.append(label_n)
                        if connected_neighbors:
                            connected_neighbors = union(connected_neighbors, make_set(label_n))
                        else:
                            connected_neighbors = make_set(label_n)

                if not connected_neighbors:
                    trees[next_label] = make_set(next_label)
                    label_map[y, x] = next_label
                    next_label += 1
                else:
                    label_map[y, x] = min(neighbor_values)
                    trees[connected_neighbors.value] = union(trees[connected_neighbors.value], connected_neighbors)

    layers = {}
    for x in range(swt.shape[1]):
        for y in range(swt.shape[0]):
            if label_map[y, x] <= 0:
                continue

            item = ld[label_map[y, x]]
            common_label = find(item).value
            label_map[y, x] = common_label

            try:
                layer = layers[common_label]
            except KeyError:
                layers[common_label] = np.zeros(
                    shape=swt.shape, dtype=np.uint16)
                layer = layers[common_label]

            layer[y, x] = 1

    logging.getLogger(__name__).info('Finished.')

    return layers, label_map
