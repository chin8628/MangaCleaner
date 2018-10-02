import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import itertools
import json
import os
import logging
from text_detection import text_detection


def main():
    clf = joblib.load('../model/0-128-manga-balanced-data.pkl')
    img_dataset_dir = '../../danbooru/resized/images/'
    connected_window_dir = '../output/connected_comp_window/'
    trained_ids = json.load(open('./trained_img_name.json', 'r'))
    test_ids = [i for i in os.listdir(connected_window_dir) if i not in trained_ids]

    for test_id in test_ids:
        src = cv2.imread(img_dataset_dir + test_id + '.jpg')
        src_gray = cv2.imread(img_dataset_dir + test_id + '.jpg', 0)
        data = text_detection(src)
        rect_list = []

        for datum in data:
            x1, y1, h, w = datum['x'], datum['y'], datum['height'], datum['width']
            x2, y2 = x1 + w, y1 + h

            isFailed = True
            while isFailed:
                roi = src_gray[y1:y2, x1:x2]
                h, w = roi.shape

                if h == 0 or w == 0:
                    break

                fd = hog(roi, pixels_per_cell=(h/10, w/10), cells_per_block=(2, 2), block_norm='L2')

                try:
                    predicted = clf.predict([fd])
                    isFailed = False
                except ValueError:
                    x1, y1, x2, y2 = x1-1, y1-1, x2+1, y2+1

            if predicted[0] and not isFailed:
                rect_list.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        chains = []
        for pair in itertools.combinations(rect_list, 2):
            rect1, rect2 = pair
            width1, width2 = rect1['x2'] - rect1['x1'], rect2['x2'] - rect2['x1']
            narrowest = min(width1, width2)
            dist = math.sqrt((rect1['x1'] - rect2['x1'])**2 + (rect1['y1'] - rect2['y1'])**2)

            if dist > narrowest * 1.5:
                continue

            added = False
            for chain in chains:
                if rect1 not in chain and rect2 in chain:
                    chain.append(rect1)
                    added = True
                elif rect1 in chain and rect2 not in chain:
                    chain.append(rect2)
                    added = True

            if not added:
                chains.append([rect1, rect2])

        mask = np.zeros(src_gray.shape, dtype=np.uint8)
        for chain in chains:
            min_x, min_y, max_x, max_y = np.inf, np.inf, -1, -1

            for rect in chain:
                min_x, min_y = min(min_x, rect['x1']), min(min_y, rect['y1'])
                max_x, max_y = max(max_x, rect['x2']), max(max_y, rect['y2'])

            mask[min_y:max_y, min_x:max_x] = 1

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(((5, 5))))
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        data = []
        index = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(src, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if w < 40 and h < 40:
                continue

            data.append({
                'id': index,
                'height': h,
                'width': w,
                'topleft_pt': {'x': x, 'y': y}
            })
            index += 1

        json.dump(data, open('../output/predicted/' + test_id + '.json', 'w'))

        # plt.imshow(src)
        # plt.show()
        # quit()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)

    main()
