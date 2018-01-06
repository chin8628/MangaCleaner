from collections import defaultdict
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.sparse, scipy.spatial
import pprint

class MangaCleaner:
    def __init__(self, path):
        src = cv2.imread(path)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        height, width = src.shape[:2]
        swt = np.zeros((height, width), np.int16)

        # Find edge
        edges = cv2.Canny(src_gray, 150, 240)

        # Find Magnitude and gradient angel
        invert_gray_image = cv2.bitwise_not(src_gray)
        sobel_x = cv2.Sobel(invert_gray_image, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(invert_gray_image, cv2.CV_64F, 0, 1)

        mag, angle = cv2.cartToPolar(sobel_x, sobel_y)

        swt = self.get_swt(edges, sobel_x, sobel_y, angle, mag, height, width)
        print('# Getting SWT is done.')
        shapes = self.connect_components(swt)
        print('# Connect component is done.')
        swts, heights, widths, topleft_pts, images = self.find_letters(swt, shapes)
        print('# Finding letters is done.')
        seperate_word_images = self.find_words(swts, heights, widths, topleft_pts, images)
        print('# Finding words is done.')

        masks =  []
        for word in seperate_word_images:
            mask = np.zeros(swt.shape, dtype=np.uint8)
            for char in word:
                mask += char
            masks.append(mask)

        bounding_points = []

        for mask in masks:
            self.showImg('Mask image', mask, 2)

            x_min_pos, x_max_pos = None, None
            y_min_pos, y_max_pos = None, None

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                min_xs = tuple(contour[contour[:,:,0].argmin()][0])[0]
                max_xs = tuple(contour[contour[:,:,0].argmax()][0])[0]
                min_ys = tuple(contour[contour[:,:,1].argmin()][0])[1]
                max_ys = tuple(contour[contour[:,:,1].argmax()][0])[1]

                if x_min_pos == None or min_xs < x_min_pos:
                    x_min_pos = min_xs
                if x_max_pos == None or max_xs > x_max_pos:
                    x_max_pos = max_xs

                if y_min_pos == None or min_xs < y_min_pos:
                    y_min_pos = min_ys
                if y_max_pos == None or max_ys > y_max_pos:
                    y_max_pos = max_ys

            print('x: ', x_min_pos, x_max_pos)
            print('y: ', y_min_pos, y_max_pos)

            bounding_points.append(((x_min_pos, y_min_pos), (x_max_pos, y_max_pos)))

        result = src.copy()

        for point in bounding_points:
            cv2.rectangle(result, point[0], point[1], (255, 0, 0), 1)

        # Display
        # self.showImg('Source image', src)
        # self.showImg('Edges image', edges, 2)
        self.showImg('Result image', result, 1)

        plt.show()

        cv2.imwrite('result.jpg', result)

    def get_swt(self, edges, sobel_x, sobel_y, angle, mag, height, width):
        swt = np.empty(edges.shape)
        swt[:] = np.Infinity

        step_x_g = sobel_x
        step_y_g = sobel_y

        np.seterr(divide='ignore', invalid='ignore')
        grad_x_g = np.divide(step_x_g, mag)
        grad_y_g = np.divide(step_y_g, mag)

        rays = []

        for y in range(height):
            for x in range(width):
                if edges[y][x] != 0:
                    cur_x, cur_y = x, y
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]

                    ray = [{'x': x, 'y': y}]
                    i = 0

                    while True:
                        i += 1
                        try:
                            cur_x = math.floor(x + grad_x * i)
                            cur_y = math.floor(y + grad_y * i)
                        except ValueError:
                            # Catch Nan value
                            break

                        try:
                            ray.append({'x': cur_x, 'y': cur_y})
                            if edges[cur_y][cur_x] != 0:
                                # Filter value which is out of domain
                                if (grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x] >= -1 and
                                    grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x] <= 1):
                                    if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                        thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                                        rays.append(ray)
                                        for pos in ray:
                                            swt[pos['y'], pos['x']] = min(thickness, swt[pos['y'], pos['x']])
                                break
                        except IndexError:
                            break
        for ray in rays:
            median = np.median([swt[pos['y'], pos['x']] for pos in ray])
            for pos in ray:
                swt[pos['y'], pos['x']] = min(median, swt[pos['y'], pos['x']])
        return swt

    def showImg(self, title, img, flag=0):
        f, ax = plt.subplots()
        f.canvas.set_window_title(title)

        if flag == 0:
            ax.imshow(img, interpolation = 'none')
        elif flag == 1:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation = 'none')
        elif flag == 2:
            ax.imshow(img, cmap='gray', interpolation = 'none')

    def connect_components(self, swt):
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
                    neighbors = [(y, x-1),   # west
                                 (y-1, x-1), # northwest
                                 (y-1, x),   # north
                                 (y-1, x+1)] # northeast
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

    def find_letters(cls, swt, shapes):
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

        return swts, heights, widths, topleft_pts, images

    def find_words(cls, swts, heights, widths, topleft_pts, images):
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
            if distance < widest * 3:
                delta_yx = topleft_pts[left] - topleft_pts[right]
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                if angle < 0:
                    angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
        atp = angle_tree.query_pairs(np.pi/12)

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
        for chain in [c for c in chains if len(c) > 3]:
            word = []
            for idx in chain:
                word.append(images[idx])
            word_images.append(word)

        return word_images
