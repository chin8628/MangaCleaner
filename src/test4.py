import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from modules.danbooru import Danbooru
from multiprocessing import Process, Manager
import itertools
from text_detection import text_detection


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def parallel(clf, features, position, window_size, callback_pos):
    for idx in tqdm(range(len(features))):
        if clf.predict([features[idx]])[0]:
            y, x = position[idx]
            callback_pos.append([y, x])


def mser(cv_image):
    vis = cv_image.copy()
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(cv_image)
    mask = np.zeros_like(cv_image)
    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)

        h, w = ymax - ymin, xmax - xmin

        if h > 80 or w > 80:
            continue

        cv2.rectangle(vis, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
        mask[ymin:ymax, xmin:xmax] += 1

    return mask


def main():
    img_color = cv2.imread('../../danbooru/resized/images/1164693.jpg')
    img = cv2.imread('../../danbooru/resized/images/1164693.jpg', 0)

    text_detection

    h, w = img.shape
    window_sizes = [25, 50]
    heatmap = np.zeros_like(img)

    mask = mser(img)

    clf = joblib.load('./model.pkl')

    for window_size in window_sizes:
        step_x = 3
        step_y = 3

        features, positions = [], []

        for y in tqdm(range(0, math.floor(h), step_y)):
            for x in range(0, w, step_x):

                total_pixel_edge = sum(sum(mask[y:y+window_size, x:x+window_size]))
                if total_pixel_edge == 0:
                    continue

                roi = img[y:y+window_size, x:x+window_size]

                if roi.shape[0] < window_size or roi.shape[1] < window_size:
                    continue

                fd = hog(roi, pixels_per_cell=(window_size/10, window_size/10), cells_per_block=(2, 2), block_norm='L2')
                features.append(fd)
                positions.append((y, x))

        processes = []
        process_number = 3
        callback_pos_block = []
        feature_chunks = chunks(features, math.ceil(len(features) / process_number))
        position_chunks = chunks(positions, math.ceil(len(positions) / process_number))

        with Manager() as manager:
            for feature_chunk, position_chunk in zip(feature_chunks, position_chunks):
                callback_pos = manager.list()
                process = Process(
                    target=parallel,
                    args=(
                        clf,
                        feature_chunk,
                        position_chunk,
                        window_size,
                        callback_pos
                    )
                )
                process.start()
                processes.append(process)
                callback_pos_block.append(callback_pos)

            for p in processes:
                p.join()

            for y, x in itertools.chain.from_iterable(callback_pos_block):
                heatmap[y: y+window_size, x: x+window_size] += 1

    rt, thr = cv2.threshold(heatmap, 8, np.amax(heatmap), cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        cv2.rectangle(img_color, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 2)

    plt.subplot(121)
    plt.imshow(heatmap)
    plt.subplot(122)
    plt.imshow(img_color)
    plt.show()


if __name__ == '__main__':
    main()
