import os
import math
import copy
from random import shuffle

import fire
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, preprocessing

from modules.file_manager import load_dataset, save_by_dict
from modules.danbooru import Danbooru


def preparing_feature(datum):
    data = copy.deepcopy(datum)

    p = [i / (data['width'] * data['height']) for i in data['hist']]

    g_mean = sum([p[i] * i for i in range(0, 256)])

    q = math.sqrt(sum([(i - g_mean)**2 * p[i] for i in range(0, 256)]))

    skew = (g_mean - list(data['hist']).index(max(data['hist']))) / q

    bin_size = 8
    new_hist = [sum(datum['hist'][i-bin_size:i]) for i in range(bin_size, 256, bin_size)]

    entropy = sum(map(lambda i: p[i] * math.log2(p[i]) if p[i] > 0 else 0, range(0, 256)))

    energy = sum([p[i]**2 for i in range(0, 256)])

    return new_hist + [skew, g_mean, np.std(data['hist']), entropy, energy]


def train(train_ids_input=None, test_ids_input=None, c_param=1, gamma_param='auto', predicted_dir='../output/predicted/'):
    if train_ids_input:
        train_ids = train_ids_input
    else:
        train_ids = [i.split('.')[0] for i in os.listdir('../output/train/')]

    shuffle(train_ids)
    train_ids = train_ids[:math.floor((len(train_ids) / 5)) * 4]
    train_dataset_files = ['../output/train/%s.json' % i for i in train_ids]
    train_data = [load_dataset(i) for i in tqdm(train_dataset_files)]

    # -------------------- FOR TESTING ----------------------- #

    if test_ids_input:
        test_ids = test_ids_input
    else:
        test_ids = list(filter(lambda x: x not in train_ids, [i.split('.')[0] for i in os.listdir('../output/test/')]))

    test_dataset_files = ['../output/test/%s.json' % i for i in test_ids]
    test_data = [load_dataset(i) for i in tqdm(test_dataset_files)]
    test_image_files = ['../../danbooru/resized/images/%s.jpg' % i for i in test_ids]

    # -------------------- END SETTING TESTING ----------------------- #

    y, x = [], []
    for idx in tqdm(range(len(train_dataset_files))):
        img_file = '../../danbooru/resized/images/' + train_ids[idx] + '.jpg'
        gray_img = cv2.imread(img_file, 0)
        balanced_data = list(filter(lambda datum: datum['is_text'] == 1, train_data[idx]))
        balanced_data += list(filter(lambda datum: datum['is_text'] == 0, train_data[idx]))[:len(balanced_data)]

        for datum in balanced_data:
            feature = [preparing_feature(datum)]
            x.append(feature)
            y.append(datum['is_text'])

    print('no. train dataset: %d' % len(x))

    scalar = preprocessing.RobustScaler().fit(x)
    x_scaled = scalar.transform(x)
    clf = svm.SVC(C=c_param, gamma=gamma_param)
    model = clf.fit(x_scaled, y)

    print(model)

    for idx in range(len(test_ids)):
        original = cv2.imread(test_image_files[idx])
        height, width = original.shape[:2]

        predicted_output = copy.deepcopy(test_data[idx])
        mask_predicted = np.zeros((height, width), np.uint8)
        for datum in predicted_output:
            x_test = [preparing_feature(datum)]
            x_test_scaled = scalar.transform(x_test)

            result = clf.predict(x_test_scaled)
            datum['is_text'] = int(result[0])

            if int(result[0]) == 1:
                y, x = datum['topleft_pt']['y'], datum['topleft_pt']['x']
                h, w = datum['height'], datum['width']
                mask_predicted[y:y + h, x:x + w] = 1

        data = []
        index = 0
        closing = cv2.morphologyEx(mask_predicted, cv2.MORPH_CLOSE, np.ones((7, 7)))
        im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < 400 or w * h > 120000:
                continue

            data.append({
                'id': index,
                'swt': -1,
                'height': h,
                'width': w,
                'topleft_pt': {'x': x, 'y': y},
                'is_text': 1,
                'hist': -1
            })

            index += 1

        save_by_dict(predicted_dir + '%s.json' % test_ids[idx], data)


if __name__ == '__main__':
    fire.Fire(train)
