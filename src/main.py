import logging
import math
from pathlib import Path
from multiprocessing import Process, Manager

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Modules
from tqdm import tqdm

from modules.label_rect import label
from text_detection import text_detection
from modules.manga109_annotation import Manga109Annotation
from modules.file_manager import save, load_dataset, save_by_dict
from modules.training import train
from modules.utils import histogram_calculate_parallel


class Main:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_labeled_data(self, page, path, labeled_file):
        # expected_height = 1200
        annotation_path = '../../Dataset_Manga/Manga109/annotations/AisazuNihaIrarenai.xml'

        self.logger.info('Annotation path: %s', path)
        self.logger.info('Absolute annotation path: %s',
                         Path(annotation_path).resolve())

        image_file = Path(str(path))
        acceptable_types = ['.jpg', '.JPG', '.jpeg', '.JPEG']

        self.logger.info('Input path: %s', path)
        self.logger.info('Absolute path: %s', image_file.resolve())

        if not image_file.is_file() or image_file.suffix not in acceptable_types:
            self.logger.error('File is not in %s types.', acceptable_types)
            quit()

        src = cv2.imread(path)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        page = int(page)

        manga109 = Manga109Annotation(annotation_path, page)
        manga109_text_area_list = manga109.get_text_area_list()

        margin = 3
        word_list = []
        fast_check_existing = {}

        for text_area in manga109_text_area_list:

            topleft_pt, bottomright_pt = text_area[0], text_area[1]
            roi = src[
                topleft_pt[0] - margin:bottomright_pt[0] + margin,
                topleft_pt[1] - margin:bottomright_pt[1] + margin
            ]

            for word in text_detection(roi, roi.shape[0]):
                key = '{}{}{}'.format(
                    word['width'], word['height'], word['topleft_pt'][0], word['topleft_pt'][1])
                if fast_check_existing.get(key, -1) != -1:
                    continue
                fast_check_existing[key] = 1
                word['is_text'] = True
                word['topleft_pt'] = (topleft_pt[0], topleft_pt[1])
                word_list.append(word)

        for word in text_detection(src, src.shape[0]):
            key = '{}{}{}'.format(
                word['width'], word['height'], word['topleft_pt'][0], word['topleft_pt'][1])
            if fast_check_existing.get(key, -1) != -1:
                continue
            fast_check_existing[key] = 1
            word['is_text'] = False
            word_list.append(word.copy())

        self.logger.info('The histogram is calculating...')

        processes = []
        hist_block = []
        process_number = 3
        len_word_list = math.ceil(len(word_list) / process_number)

        with Manager() as manager:
            for idx in range(0, process_number):
                hist = manager.list()
                process = Process(
                    target=histogram_calculate_parallel,
                    args=(
                        src_gray, word_list[len_word_list * idx: len_word_list * (idx + 1)], hist)
                )
                process.start()
                processes.append(process)
                hist_block.append(hist)

            for process in processes:
                process.join()

            index = 0
            for hist_list in hist_block:
                for hist in list(hist_list):
                    word = word_list[index]
                    word['hist'] = hist.copy()

                    index += 1

        swts, heights, widths, topleft_pts, is_texts, hists = [], [], [], [], [], []
        for word in word_list:
            swts.append(word['swt'])
            heights.append(word['height'])
            widths.append(word['width'])
            topleft_pts.append(word['topleft_pt'])
            is_texts.append(word['is_text'])
            hists.append(word['hist'])

        save(labeled_file, swts, heights, widths, topleft_pts, is_texts, hists)

        return 0

    def plot_from_files(self, *paths):
        swts, widths, heights, diameters, hw_ratios, is_texts = [], [], [], [], [], []

        for path in paths:
            self.logger.info('%s is Loading...', path)
            data = load_dataset(path)

            for datum in data:
                swts.append(datum['swt'])
                widths.append(datum['width'])
                heights.append(datum['height'])
                diameters.append(
                    math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height']))
                hw_ratios.append(datum['height'] / datum['width'])
                is_texts.append(datum['is_text'])

        swts, widths, heights, diameters, = np.array(swts), np.array(
            widths), np.array(heights), np.array(diameters)
        hw_ratios, is_texts = np.array(hw_ratios), np.array(is_texts)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(swts[is_texts == 0], diameters[is_texts == 0],
                   hw_ratios[is_texts == 0], s=4)
        ax.scatter(swts[is_texts == 1], diameters[is_texts == 1],
                   hw_ratios[is_texts == 1], s=4)
        ax.set_xlabel('swt')
        ax.set_ylabel('diameter')
        ax.set_zlabel('h/w ratio')
        plt.show()

    @staticmethod
    def draw_rect(img_path, dataset_path):
        data = load_dataset(dataset_path)
        src = cv2.imread(img_path)

        # widths, heights, topleft_pts = [], [], []
        # for datum in list(filter(lambda x: x['is_text'] == 0, data)):
        #     topleft_pts.append(
        #         (datum['topleft_pt']['y'], datum['topleft_pt']['x']))
        #     widths.append(datum['width'])
        #     heights.append(datum['height'])

        # label(src, topleft_pts, heights, widths, (0, 0, 255))

        widths, heights, topleft_pts = [], [], []
        for datum in list(filter(lambda x: x['is_text'] == 1, data)):
            topleft_pts.append(
                (datum['topleft_pt']['y'], datum['topleft_pt']['x']))
            widths.append(datum['width'])
            heights.append(datum['height'])

        label(src, topleft_pts, heights, widths, (255, 0, 0))

    def svm(self):
        data = list()

        data.append(load_dataset('../output/verified/Aisazu-006.json'))
        data.append(load_dataset('../output/verified/Aisazu-014.json'))
        data.append(load_dataset('../output/verified/Aisazu-023.json'))
        data.append(load_dataset('../output/verified/Aisazu-026.json'))
        data.append(load_dataset('../output/verified/Aisazu-035.json'))

        self.logger.info('Data is preparing...')
        trains = [
            {'x': [], 'y': []},
            {'x': [], 'y': []},
            {'x': [], 'y': []},
            {'x': [], 'y': []}
        ]

        for dataIdx in range(4):
            for datum in data[dataIdx]:
                diameter = math.sqrt(
                    datum['width'] * datum['width'] + datum['height'] * datum['height'])
                feature = datum['hist'] + [datum['swt']]
                trains[dataIdx]['x'].append(feature)
                trains[dataIdx]['y'].append(datum['is_text'])

        y, x = [], []
        for idx in range(4):
            len_y_true = sum(trains[idx]['y'])
            x += trains[idx]['x'][:len_y_true * 2]
            y += trains[idx]['y'][:len_y_true * 2]

        print('count data: {}'.format(len(y)))

        x_test, y_test = [], []
        for datum in tqdm(data[4]):
            diameter = math.sqrt(
                datum['width'] * datum['width'] + datum['height'] * datum['height'])
            feature = datum['hist'] + [datum['swt']]
            x_test.append(feature)
            y_test.append(datum['is_text'])

        result = train(x, y, x_test, y_test)

        predicted_output = []
        for index in range(0, len(data[4])):
            predict_data = data[4][index].copy()
            predict_data['is_text'] = int(result[index])
            predicted_output.append(predict_data)

        save_by_dict('../output/predicted/Aisazu-035.json', predicted_output)


if __name__ == '__main__':
    fire.Fire(Main)
