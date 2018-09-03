import os
import math
import copy

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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


def svm():
    dataset_filenames = [
        '../output/verified/Aisazu-006.json',
        '../output/verified/Aisazu-014.json',
        '../output/verified/Aisazu-023.json',
        '../output/verified/Aisazu-026.json',
        '../output/verified/Aisazu-035.json',
    ]
    data = [load_dataset(i) for i in tqdm(dataset_filenames)]
    img_filenames = [
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/006.jpg',
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/014.jpg',
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/023.jpg',
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/026.jpg',
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/035.jpg',
    ]

    # -------------------- FOR SMALL TESTING ----------------------- #

    dataset_filenames_test = [
        '../output/test/AisazuNihaIrarenai-034.jpg.json'
    ]
    data_test = [load_dataset(i) for i in tqdm(dataset_filenames_test)]
    img_filenames_test = [
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/034.jpg',
    ]
    annotation_path = ['../../Dataset_Manga/Manga109/annotations/AisazuNihaIrarenai.xml']
    page_number = [34]

    # -------------------- FOR GIANT TESTING ----------------------- #

    # dataset_filenames_test = [i for i in os.listdir('../output/test/')]
    # img_filenames_test = [
    #     '../../Dataset_Manga/Manga109/images/' + '.'.join(i.replace('-', '/').split('.')[:-1])
    #     for i in dataset_filenames_test
    # ]
    # data_test = [load_dataset('../output/test/' + i) for i in tqdm(dataset_filenames_test)]
    # annotation_path = [
    #     '../../Dataset_Manga/Manga109/annotations/' + i.split('-')[0] + '.xml'
    #     for i in dataset_filenames_test
    # ]
    # page_number = [int(i.split('.')[:-2][0][-3:].lstrip('0')) for i in dataset_filenames_test]

    # -------------------- END SETTING TESTING ----------------------- #

    y, x = [], []
    for idx in range(len(dataset_filenames)):
        x_uncut, y_uncut = [], []

        for datum in data[idx]:
            feature = preparing_feature(datum)
            x_uncut.append(feature)
            y_uncut.append(datum['is_text'])

        count_true = sum(filter(lambda y: y == 1, y_uncut))
        x += x_uncut[:count_true * 2]
        y += y_uncut[:count_true * 2]

    print('count data: {}'.format(len(y)))

    x_test = []
    for idx in range(len(dataset_filenames_test)):
        for datum in data_test[idx]:
            feature = preparing_feature(datum)
            x_test.append(feature)

    result = train(x, y, x_test)

    tp, fp, tn, fn = 0, 0, 0, 0

    for idx in range(0, len(dataset_filenames_test)):
        img = cv2.imread(img_filenames_test[idx], 0)
        height, width = img.shape
        data = data_test[idx]

        predicted_output = []
        for index in range(0, len(data)):
            predict_data = copy.deepcopy(data[index])
            predict_data['is_text'] = int(result[index])
            predicted_output.append(predict_data)

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
