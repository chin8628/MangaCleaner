from collections import defaultdict
import numpy as np

def connect_components(swt):
    class Label(object):
        def __init__(self, value):
            self.value = value
            self.parent = self
            self.rank = 0

        def __eq__(self, other):
            if type(other) is type(self):
                return self.value == other.value
            else:
                return False

        def __ne__(self, other):
            return not self.__eq__(other)

    ld = {}

    def MakeSet(x):
        try:
            return ld[x]
        except KeyError:
            item = Label(x)
            ld[x] = item
            return item

    def Find(item):
        if item.parent != item:
            item.parent = Find(item.parent)
        return item.parent

    def Union(x, y):
        """
        :param x:
        :param y:
        :return: root node of new union tree
        """
        x_root = Find(x)
        y_root = Find(y)
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
            if sw_point < np.Infinity and sw_point > 0:
                neighbors = [(y, x - 1),  # west
                             (y - 1, x - 1),  # northwest
                             (y - 1, x),  # north
                             (y - 1, x + 1)]  # northeast
                connected_neighbors = None
                neighborvals = []

                for neighbor in neighbors:
                    try:
                        sw_n = swt[neighbor]
                        label_n = label_map[neighbor]
                    except IndexError:
                        continue
                    if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                        neighborvals.append(label_n)
                        if connected_neighbors:
                            connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                        else:
                            connected_neighbors = MakeSet(label_n)

                if not connected_neighbors:
                    trees[next_label] = (MakeSet(next_label))
                    label_map[y, x] = next_label
                    next_label += 1
                else:
                    label_map[y, x] = min(neighborvals)
                    trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

    layers = {}
    contours = defaultdict(list)
    for x in range(swt.shape[1]):
        for y in range(swt.shape[0]):
            if label_map[y, x] > 0:
                item = ld[label_map[y, x]]
                common_label = Find(item).value
                label_map[y, x] = common_label
                contours[common_label].append([x, y])
                try:
                    layer = layers[common_label]
                except KeyError:
                    layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                    layer = layers[common_label]

                layer[y, x] = 1
    return layers
