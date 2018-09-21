import os
import cv2
import fire
import numpy as np
import matplotlib.pyplot as plt

# Modules
from tqdm import tqdm
import cv2

from modules.danbooru import Danbooru
from modules.file_manager import load_dataset


def isRectMatched(single_mask_truth, full_mask_predicted, threashold):
    single_mask_truth_area = sum(sum(single_mask_truth))
    truth_n_predicted = single_mask_truth + full_mask_predicted
    overlaped_area = sum(sum(truth_n_predicted == 2))

    h, w = single_mask_truth.shape
    output = np.zeros((h+2, w+2), np.uint8)
    nz = np.nonzero(truth_n_predicted > 1)

    if len(nz[1]) == 0:
        return False

    flood_fill = cv2.floodFill((truth_n_predicted > 0).astype(np.uint8), output, (nz[1][0], nz[0][0]), 1)[2]
    flood_fill = flood_fill[1:-1, 1:-1]
    single_mask_predicted = flood_fill + full_mask_predicted > 1
    single_mask_predicted_area = sum(sum(single_mask_predicted))

    ratio_overlap_truth = overlaped_area / single_mask_truth_area
    ratio_overlap_predicted = overlaped_area / single_mask_predicted_area

    if ratio_overlap_truth > threashold and ratio_overlap_predicted > threashold:
        return True
    else:
        return False


def evaluate():
    matched_rect = 0
    no_rect_in_dataset, no_detected_rect = 0, 0

    test_ids = [i.split('.')[0] for i in os.listdir('../output/predicted/')]
    test_image_files = ['../../danbooru/resized/images/%s.jpg' % i for i in test_ids]
    test_dataset_files = ['../output/predicted/%s.json' % i for i in test_ids]
    test_data = [load_dataset(i) for i in tqdm(test_dataset_files)]

    for test_datum in tqdm(test_data):
        index = test_data.index(test_datum)
        original = cv2.imread(test_image_files[index])
        height, width = original.shape[:2]

        mask_predicted = np.zeros((height, width))
        for datum in filter(lambda x: x['is_text'] == 1, test_datum):
            y, x = datum['topleft_pt']['y'], datum['topleft_pt']['x']
            mask_predicted[y:y + datum['height'], x:x + datum['width']] = 1

        text_area_dataset = Danbooru(test_ids[index]).get_text_area()

        no_detected_rect += len(list(filter(lambda x: x['is_text'] == 1, test_datum)))
        no_rect_in_dataset = len(text_area_dataset)

        for text_area in text_area_dataset:
            x1, y1, w, h = text_area['x'], text_area['y'], text_area['width'], text_area['height']
            x2, y2 = x1 + w, y1 + h

            mask_truth = np.zeros((height, width))
            mask_truth[y1:y2, x1:x2] = 1

            if isRectMatched(mask_truth, mask_predicted, 0.5):
                matched_rect += 1

    recall = matched_rect / no_rect_in_dataset
    precision = matched_rect / no_detected_rect

    print('No. matched rect:', matched_rect)
    print('No. rect in dataset:', no_rect_in_dataset)
    print('No. detected rect:', no_detected_rect)
    print('P: {} R: {}'.format(precision, recall))
    print('F-measure: {}'.format(round(2 * ((precision * recall) / (precision + recall)), 4)))

    if np.isnan(round(2 * ((precision * recall) / (precision + recall)), 4)):
        return 0
    else:
        return round(2 * ((precision * recall) / (precision + recall)), 4)


if __name__ == '__main__':
    fire.Fire(evaluate)
