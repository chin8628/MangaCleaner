import os
import cv2
import fire
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# Modules
from tqdm import tqdm
import cv2

from modules.manga109 import Manga109
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


def evaluate():
    matched_rect = 0
    no_rect_in_dataset, no_detected_rect = 0, 0
    predicted_dir = '../output/predicted/'

    for title in os.listdir(predicted_dir):
        manga109 = Manga109(title)

        for page_id in [int(i.split('.')[0]) for i in os.listdir(predicted_dir + title)]:
            rect_truths = []
            text_area_dataset = list(manga109.get_text_area(page_id))
            test_data = load_dataset(predicted_dir + title + '/%03d.json' % page_id)

            for text_area in text_area_dataset:
                x1, y1, w, h = text_area['x'], text_area['y'], text_area['width'], text_area['height']
                x2, y2 = x1 + w, y1 + h
                rect_truths.append(Rect(x1=x1, y1=y1, x2=x2, y2=y2))

            for datum in test_data:
                y1, x1 = datum['topleft_pt']['y'], datum['topleft_pt']['x']
                h, w = datum['height'], datum['width']
                y2, x2 = y1 + h, x1 + w

                rect_test = Rect(x1=x1, y1=y1, x2=x2, y2=y2)
                for rect_truth in rect_truths:
                    if isRectMatched(rect_truth, rect_test, 0.5):
                        matched_rect += 1
                        break

            no_detected_rect += len(test_data)
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
