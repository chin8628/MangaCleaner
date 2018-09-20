import os
import math
import copy

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, preprocessing

from modules.file_manager import load_dataset, save_by_dict
from modules.manga109_annotation import Manga109Annotation


def isRectMatched(mask_truth, mask_predicted, threasholdRect):
    mask_truth_area = sum(sum(mask_truth))
    truth_n_predicted = mask_truth + mask_predicted
    overlaped_area = sum(sum(truth_n_predicted == 2))

    h, w = mask_truth.shape
    output = np.zeros((h+2, w+2), np.uint8)
    nz = np.nonzero(truth_n_predicted > 1)

    if len(nz[1]) == 0:
        return False

    flood_fill = cv2.floodFill((truth_n_predicted > 0).astype(np.uint8), output, (nz[1][0], nz[0][0]), 1)[2]
    flood_fill = flood_fill[1:-1, 1:-1]
    single_mask_predicted = flood_fill + mask_predicted > 1
    single_mask_predicted_area = sum(sum(single_mask_predicted))

    ratio_overlap_truth = overlaped_area / mask_truth_area
    ratio_overlap_predicted = overlaped_area / single_mask_predicted_area

    if ratio_overlap_truth > threasholdRect and ratio_overlap_predicted > threasholdRect:
        return True
    else:
        return False


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
        '../output/verified/Aisazu-006.json',
        '../output/verified/Aisazu-014.json',
        '../output/verified/Aisazu-023.json',
        '../output/verified/Aisazu-026.json',
        '../output/verified/Aisazu-035.json',
        '../output/verified/AosugiruHaru-010.json',
        '../output/verified/AosugiruHaru-011.json',
        '../output/verified/AosugiruHaru-017.json',
        '../output/verified/AosugiruHaru-018.json',
        '../output/verified/AosugiruHaru-021.json',
        '../output/verified/Arisa-013.json',
        '../output/verified/Arisa-016.json',
        '../output/verified/Arisa-017.json',
        '../output/verified/Arisa-024.json',
        '../output/verified/Arisa-041.json',
        '../output/verified/BakuretsuKungFuGirl-004.json',
        '../output/verified/BakuretsuKungFuGirl-012.json',
        '../output/verified/BakuretsuKungFuGirl-017.json',
        '../output/verified/BakuretsuKungFuGirl-021.json',
        '../output/verified/BakuretsuKungFuGirl-023.json',
        '../output/verified/DollGun-004.json',
        '../output/verified/DollGun-008.json',
        '../output/verified/DollGun-013.json',
        '../output/verified/DollGun-028.json',
        '../output/verified/DollGun-031.json',
        '../output/verified/EvaLady-043.json',
        '../output/verified/LoveHina_vol01-025.json'
    ]
    data = [load_dataset(i) for i in tqdm(dataset_filenames)]

    # -------------------- FOR SMALL TESTING ----------------------- #

    dataset_filenames_test = [
        '../output/AisazuNihaIrarenai-034.json'
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

    matched_rect, non_matched_rect = 0, 0
    no_rect_in_dataset, no_detected_rect = 0, 0

    for idx in tqdm(range(len(dataset_filenames_test))):
        original = cv2.imread(img_filenames_test[idx])
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

        mask_predicted = np.zeros((height, width), np.uint8)
        for datum in filter(lambda x: x['is_text'] == 1, predicted_output):
            y, x = datum['topleft_pt']['y'], datum['topleft_pt']['x']
            mask_predicted[y:y + datum['height'], x:x + datum['width']] = 1

        pre_mask_predicted = np.zeros(mask_predicted.shape, np.uint8)
        while (mask_predicted != pre_mask_predicted).any():
            pre_mask_predicted = copy.deepcopy(mask_predicted)

            closing = cv2.morphologyEx(pre_mask_predicted, cv2.MORPH_CLOSE, np.ones((7, 7)))
            im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_predicted = np.zeros(mask_predicted.shape, np.uint8)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                mask_predicted[y:y + h, x:x + w] = 1

            # plt.figure(1)
            # plt.subplot(211)
            # plt.imshow(pre_mask_predicted)
            # plt.subplot(212)
            # plt.imshow(mask_predicted)
            # plt.show()

        im2, contours, hierarchy = cv2.findContours(mask_predicted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 2)

        no_detected_rect += len(contours)

        manga109_text_area_list = Manga109Annotation(annotation_path[idx], page_number[idx]).get_text_area_list()
        no_rect_in_dataset += len(manga109_text_area_list)
        for text_area in manga109_text_area_list:
            mask_truth = np.zeros((height, width), np.int8)
            x1, y1, x2, y2 = text_area[0][1], text_area[0][0], text_area[1][1], text_area[1][0]
            mask_truth[y1:y2, x1:x2] = 1

            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if isRectMatched(mask_truth, mask_predicted, 0.5):
                matched_rect += 1
            else:
                non_matched_rect += 1

        plt.imshow(original)
        plt.show()

    recall = matched_rect / no_rect_in_dataset
    precision = matched_rect / no_detected_rect

    print('No. matched rect:', matched_rect)
    print('No. rect in dataset:', no_rect_in_dataset)
    print('No. detected rect:', no_detected_rect)
    print('P: {} R: {}'.format(precision, recall))
    print('F-measure: {}'.format(round(2 * ((precision * recall) / (precision + recall)), 4)))

    # save_by_dict('../output/predicted/{}-{}.json'.format(manga_name, output_filename), predicted_output)

    if np.isnan(round(2 * ((precision * recall) / (precision + recall)), 4)):
        return 0
    else:
        return round(2 * ((precision * recall) / (precision + recall)), 4)
