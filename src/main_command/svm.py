import os
import math
import copy

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, preprocessing

from modules.training import train
from modules.file_manager import load_dataset, save_by_dict
from modules.manga109_annotation import Manga109Annotation


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


def train():
    dataset_filenames = [
        '../output/train/AosugiruHaru-005.json',
        '../output/train/AosugiruHaru-008.json',
        '../output/train/AosugiruHaru-015.json',
        '../output/train/AosugiruHaru-018.json',
        '../output/train/AosugiruHaru-079.json',
        '../output/train/Arisa-013.json',
        '../output/train/Arisa-025.json',
        '../output/train/Arisa-026.json',
        '../output/train/Arisa-040.json',
        '../output/train/Arisa-041.json',
        '../output/train/BakuretsuKungFuGirl-012.json',
        '../output/train/BakuretsuKungFuGirl-013.json',
        '../output/train/BakuretsuKungFuGirl-016.json',
        '../output/train/BakuretsuKungFuGirl-017.json',
        '../output/train/BakuretsuKungFuGirl-021.json',
        '../output/train/DollGun-007.json',
        '../output/train/DollGun-008.json',
        '../output/train/DollGun-010.json',
        '../output/train/DollGun-017.json',
        '../output/train/DollGun-026.json',
        '../output/train/DollGun-027.json',
        '../output/train/DollGun-028.json',
        '../output/train/EvaLady-009.json',
        '../output/train/EvaLady-011.json',
        '../output/train/EvaLady-016.json',
        '../output/train/EvaLady-027.json',
        '../output/train/EvaLady-043.json',
        '../output/train/LoveHina_vol01-005.json',
        '../output/train/LoveHina_vol01-006.json',
        '../output/train/LoveHina_vol01-010.json',
        '../output/train/LoveHina_vol01-014.json',
        '../output/train/LoveHina_vol01-016.json',
        '../output/train/LoveHina_vol01-021.json',
        '../output/train/LoveHina_vol01-025.json'
    ]
    data = [load_dataset(i) for i in tqdm(dataset_filenames)]

    # -------------------- FOR SMALL TESTING ----------------------- #

    # dataset_filenames_test = [
    #     '../output/test/AisazuNihaIrarenai-034.jpg.json'
    # ]
    # data_test = [load_dataset(i) for i in tqdm(dataset_filenames_test)]
    # img_filenames_test = [
    #     '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/034.jpg',
    # ]
    # annotation_path = ['../../Dataset_Manga/Manga109/annotations/AisazuNihaIrarenai.xml']
    # page_number = [34]

    # -------------------- FOR GIANT TESTING ----------------------- #

    dataset_filenames_test = [i for i in os.listdir('../output/test/')]
    img_filenames_test = [
        '../../Dataset_Manga/Manga109/images/' + '.'.join(i.replace('-', '/').split('.')[:-1])
        for i in dataset_filenames_test
    ]
    data_test = [load_dataset('../output/test/' + i) for i in tqdm(dataset_filenames_test)]
    annotation_path = [
        '../../Dataset_Manga/Manga109/annotations/' + i.split('-')[0] + '.xml'
        for i in dataset_filenames_test
    ]
    page_number = [int(i.split('.')[:-2][0][-3:].lstrip('0')) for i in dataset_filenames_test]

    # -------------------- END SETTING TESTING ----------------------- #

    y, x = [], []
    for idx in range(len(dataset_filenames)):
        balanced_data = list(filter(lambda datum: datum['is_text'] == 1, data[idx]))
        balanced_data += list(filter(lambda datum: datum['is_text'] == 0, data[idx]))[:len(balanced_data)]

        for datum in balanced_data:
            feature = preparing_feature(datum)
            x.append(feature)
            y.append(datum['is_text'])

    scalar = preprocessing.RobustScaler().fit(x)
    x_scaled = scalar.transform(x)
    clf = svm.SVC(C=10**1.18)
    model = clf.fit(x_scaled, y)

    print(model)

    print('count data: {}'.format(len(y)))

    tp, fp, tn, fn = 0, 0, 0, 0

    for idx in range(len(dataset_filenames_test)):
        img = cv2.imread(img_filenames_test[idx], 0)
        height, width = img.shape

        predicted_output = copy.deepcopy(data_test[idx])

        for datum in predicted_output:
            x_test = [preparing_feature(datum)]
            x_test_scaled = scalar.transform(x_test)

            result = clf.predict(x_test_scaled)
            datum['is_text'] = int(result[0])

        manga_name = img_filenames_test[idx].split('/')[-2]
        output_filename = img_filenames_test[idx].split('/')[-1].split('.')[0]
        save_by_dict('../output/predicted/{}-{}.json'.format(manga_name, output_filename), predicted_output)

        manga109_text_area_list = Manga109Annotation(annotation_path[idx], page_number[idx]).get_text_area_list()
        mask_truth = np.zeros((height, width), np.int64)
        for text_area in manga109_text_area_list:
            topleft_pt, bottomright_pt = text_area[0], text_area[1]
            mask_truth[topleft_pt[0]:bottomright_pt[0], topleft_pt[1]:bottomright_pt[1]] = 1

        mask_predicted = np.zeros((height, width), np.int64)
        for datum in filter(lambda x: x['is_text'] == 1, predicted_output):
            topleft_pt = datum['topleft_pt']
            mask_predicted[
                topleft_pt['y']:topleft_pt['y'] + datum['height'],
                topleft_pt['x']:topleft_pt['x'] + datum['width']
            ] = 1

        tp += sum(sum(np.bitwise_and(mask_truth, mask_predicted)))
        fp += sum(sum((mask_predicted - mask_truth) == 1))
        tn += sum(sum((mask_truth + mask_predicted) == 0))
        fn += sum(sum((mask_truth - mask_predicted) == 1))

    print('TP: {} FP: {} TN: {} FN: {}'.format(tp, fp, tn, fn))

    try:
        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        print('P: {} R: {}'.format(precision, recall))
        print('F-measure: {}'.format(round(2 * ((precision * recall) / (precision + recall)), 4)))
    except ZeroDivisionError:
        print('Divided by zero')

    if np.isnan(round(2 * ((precision * recall) / (precision + recall)), 4)):
        return 0
    else:
        return round(2 * ((precision * recall) / (precision + recall)), 4)
