import os
import cv2
import fire
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# Modules
from tqdm import tqdm
import cv2

from modules.danbooru import Danbooru
from modules.file_manager import load_dataset

Rect = namedtuple('Rect', 'x1 y1 x2 y2')


def isRectMatched(rect_truth: Rect, rect_test: Rect, threashold: float):
    dx = min(rect_truth.x2, rect_test.x2) - max(rect_truth.x1, rect_test.x1)
    dy = min(rect_truth.y2, rect_test.y2) - max(rect_truth.y1, rect_test.y1)

    if dx <= 0 or dy <= 0:
        return False

    overlaped_area = dx * dy
    truth_area = (rect_truth.x2 - rect_truth.x1) * (rect_truth.y2 - rect_truth.y1)
    predicted_area = (rect_test.x2 - rect_test.x1) * (rect_test.y2 - rect_test.y1)

    ratio_overlap_truth = overlaped_area / truth_area
    ratio_overlap_predicted = overlaped_area / predicted_area

    if ratio_overlap_predicted > 1 or ratio_overlap_truth > 1:
        print('Error! ratio > 1 (overlap/predicted: %f, overlap/truth: %f)' %
              (ratio_overlap_predicted, ratio_overlap_truth))
        quit()

    if ratio_overlap_truth > threashold and ratio_overlap_predicted > threashold:
        return True
    else:
        return False


def evaluate(test_ids_input=None, predicted_dir='../output/predicted/'):
    matched_rect = 0
    no_rect_in_dataset, no_detected_rect = 0, 0

    if test_ids_input:
        test_ids = test_ids_input
    else:
        test_ids = [i.split('.')[0] for i in os.listdir(predicted_dir) if len(i.split('.')) == 2]

    test_image_files = ['../../danbooru/resized/images/%s.jpg' % i for i in test_ids]
    test_dataset_files = [predicted_dir + '%s.json' % i for i in test_ids]
    test_data = [load_dataset(i) for i in tqdm(test_dataset_files)]

    for test_datum in tqdm(test_data):
        index = test_data.index(test_datum)
        original = cv2.imread(test_image_files[index])
        height, width = original.shape[:2]

        rect_truths = []
        text_area_dataset = Danbooru(test_ids[index]).get_text_area()
        for text_area in text_area_dataset:
            x1, y1, w, h = text_area['x'], text_area['y'], text_area['width'], text_area['height']
            x2, y2 = x1 + w, y1 + h
            rect_truths.append(Rect(x1=x1, y1=y1, x2=x2, y2=y2))

        for datum in filter(lambda x: x['is_text'] == 1, test_datum):
            y1, x1 = datum['topleft_pt']['y'], datum['topleft_pt']['x']
            h, w = datum['height'], datum['width']
            y2, x2 = y1 + h, x1 + w

            rect_test = Rect(x1=x1, y1=y1, x2=x2, y2=y2)
            for rect_truth in rect_truths:
                if isRectMatched(rect_truth, rect_test, 0.5):
                    matched_rect += 1
                    break

        no_detected_rect += len(list(filter(lambda x: x['is_text'] == 1, test_datum)))
        no_rect_in_dataset += len(text_area_dataset)

    recall = matched_rect / no_rect_in_dataset
    precision = matched_rect / no_detected_rect

    try:
        fmeasure = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        fmeasure = 0

    print('No. matched rect:', matched_rect)
    print('No. rect in dataset:', no_rect_in_dataset)
    print('No. detected rect:', no_detected_rect)
    print('P: {} R: {} F: {}'.format(precision, recall, fmeasure))

    return {'r': recall, 'p': precision, 'f': fmeasure}


if __name__ == '__main__':
    fire.Fire(evaluate)
