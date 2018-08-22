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
    ]

    dataset_filenames_test = [
        '../output/verified/Aisazu-035.json',
    ]

    img_filenames_test = [
        '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/035.jpg',
    ]

    data = [load_dataset(i) for i in dataset_filenames]
    data_test = [load_dataset(i) for i in dataset_filenames_test]

    annotation_path = '../../Dataset_Manga/Manga109/annotations/AisazuNihaIrarenai.xml'
    page_number = [35]

    trains = [
        {'x': [], 'y': []},
        {'x': [], 'y': []},
        {'x': [], 'y': []},
        {'x': [], 'y': []}
    ]

    y, x = [], []
    for idx in range(len(dataset_filenames)):
        for datum in data[idx]:
            feature = datum['hist'] + [datum['swt']]
            trains[idx]['x'].append(feature)
            trains[idx]['y'].append(datum['is_text'])

        len_y_true = sum(trains[idx]['y'])
        x += trains[idx]['x'][:len_y_true * 2]
        y += trains[idx]['y'][:len_y_true * 2]

    print('count data: {}'.format(len(y)))

    x_test, y_test = [], []
    for idx in range(len(dataset_filenames_test)):
        for datum in data[idx]:
            feature = datum['hist'] + [datum['swt']]
            x_test.append(feature)
            y_test.append(datum['is_text'])

    result = train(x, y, x_test, y_test)

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

            if result[index] == 0:
                predict_data['is_text'] = int(result[index])
                predicted_output.append(predict_data)
            else:
                ret, thresh = cv2.threshold(sub_area_img, 230, 255, cv2.THRESH_BINARY)

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
            print('P: {} R: {}'.format(round(tp / (tp + fp), 4), round(tp / (tp + fn), 4)))
        except ZeroDivisionError:
            print('Divided by zero')
