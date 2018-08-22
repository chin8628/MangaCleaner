import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from modules.training import train
from modules.file_manager import save, load_dataset, save_by_dict
from modules.manga109_annotation import Manga109Annotation


def svm():
    dataset_filenames = [
        '../output/verified/Aisazu-006.json',
        '../output/verified/Aisazu-014.json',
        '../output/verified/Aisazu-023.json',
        '../output/verified/Aisazu-026.json',
        '../output/verified/Aisazu-035.json',
    ]

    dataset_filenames_test = [
        '../output/raw/AppareKappore-005.json'
    ]

    img_filenames_test = [
        '../../Dataset_Manga/Manga109/images/AppareKappore/005.jpg',
    ]

    data = [load_dataset(i) for i in dataset_filenames]
    data_test = [load_dataset(i) for i in dataset_filenames_test]

    annotation_path = '../../Dataset_Manga/Manga109/annotations/AppareKappore.xml'
    page_number = [5]

    y, x = [], []
    for idx in range(len(dataset_filenames)):
        x_uncut, y_uncut = [], []

        for datum in data[idx]:
            feature = datum['hist'] + [datum['swt']]
            x_uncut.append(feature)
            y_uncut.append(datum['is_text'])

        count_true = sum(filter(lambda y: y == 1, y_uncut))
        x += x_uncut[:count_true * 2]
        y += y_uncut[:count_true * 2]

    print('count data: {}'.format(len(y)))

    x_test = []
    for idx in range(len(dataset_filenames_test)):
        for datum in data_test[idx]:
            feature = datum['hist'] + [datum['swt']]
            x_test.append(feature)

    result = train(x, y, x_test)

    for idx in range(0, len(dataset_filenames_test)):
        img = cv2.imread(img_filenames_test[idx], 0)
        height, width = img.shape
        data = data_test[idx]

        predicted_output = []
        for index in range(0, len(data)):
            predict_data = data[index].copy()
            sub_area_img = img[
                predict_data['topleft_pt']['y']:predict_data['topleft_pt']['y'] + predict_data['height'],
                predict_data['topleft_pt']['x']:predict_data['topleft_pt']['x'] + predict_data['width']
            ]
            sub_height, sub_width = sub_area_img.shape

            # predict_data['is_text'] = int(result[index])
            # predicted_output.append(predict_data)

            if result[index] == 0:
                predict_data['is_text'] = int(result[index])
                predicted_output.append(predict_data)
            else:
                ret, thresh = cv2.threshold(sub_area_img, 200, 255, cv2.THRESH_BINARY)

                if sum(sum(thresh == 255)) / (sub_height*sub_width) > 0.4:
                    predicted_output.append(predict_data)
                else:
                    result[index] = 0
                    predict_data['is_text'] = int(result[index])
                    predicted_output.append(predict_data)

        manga_name = img_filenames_test[idx].split('/')[-2]
        output_filename = img_filenames_test[idx].split('/')[-1].split('.')[0]
        save_by_dict('../output/predicted/{}-{}.json'.format(manga_name, output_filename), predicted_output)

        manga109_text_area_list = Manga109Annotation(annotation_path, page_number[idx]).get_text_area_list()

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

        tp = sum(sum(np.bitwise_and(mask_truth, mask_predicted)))
        fp = sum(sum((mask_predicted - mask_truth) == 1))
        tn = sum(sum((mask_truth + mask_predicted) == 0))
        fn = sum(sum((mask_truth - mask_predicted) == 1))

        print('TP: {} FP: {} TN: {} FN: {}'.format(tp, fp, tn, fn))

        try:
            precision = round(tp / (tp + fp), 4)
            recall = round(tp / (tp + fn), 4)
            print('P: {} R: {}'.format(precision, recall))
            print('F-measure: {}'.format(round(2 * ((precision * recall) / (precision + recall)), 4)))
        except ZeroDivisionError:
            print('Divided by zero')
