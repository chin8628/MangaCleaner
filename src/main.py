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
from modules.file_manager import save, load_dataset
from modules.training import train
from modules.utils import histogram_calculate_parallel


class Main:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_labeled_data(self, page, path, labeled_file):
        # expected_height = 1200
        annotation_path = '../../Dataset_Manga/Manga109/annotations/GOOD_KISS_Ver2.xml'

        self.logger.info('Annotation path: %s', path)
        self.logger.info('Absolute annotation path: %s', Path(annotation_path).resolve())

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

        margin = 10
        word_list = []
        fast_check_existing = {}

        for text_area in manga109_text_area_list:

            topleft_pt, bottomright_pt = text_area[0], text_area[1]
            roi = src[
                  topleft_pt[0] - margin:bottomright_pt[0] + margin,
                  topleft_pt[1] - margin:bottomright_pt[1] + margin
                  ]

            cv2.imshow('test', roi)
            cv2.waitKey(0)

            for word in text_detection(roi, roi.shape[0]):
                key = '{}{}{}'.format(word['width'], word['height'], word['topleft_pt'][0], word['topleft_pt'][1])
                if fast_check_existing.get(key, -1) != -1:
                    continue
                fast_check_existing[key] = 1
                word['is_text'] = True
                word['topleft_pt'] = (topleft_pt[0], topleft_pt[1])
                word_list.append(word)

        for word in text_detection(src, src.shape[0]):
            key = '{}{}{}'.format(word['width'], word['height'], word['topleft_pt'][0], word['topleft_pt'][1])
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
                    args=(src_gray, word_list[len_word_list * idx: len_word_list * (idx + 1)], hist)
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

    def plot_from_files(self, *paths):
        swts, widths, heights, diameters, hw_ratios, is_texts = [], [], [], [], [], []

        for path in paths:
            self.logger.info('%s is Loading...', path)
            data = load_dataset(path)

            for datum in data:
                swts.append(datum['swt'])
                widths.append(datum['width'])
                heights.append(datum['height'])
                diameters.append(math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height']))
                hw_ratios.append(datum['height'] / datum['width'])
                is_texts.append(datum['is_text'])

        swts, widths, heights, diameters, = np.array(swts), np.array(widths), np.array(heights), np.array(diameters)
        hw_ratios, is_texts = np.array(hw_ratios), np.array(is_texts)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(swts[is_texts == 0], diameters[is_texts == 0], hw_ratios[is_texts == 0], s=4)
        ax.scatter(swts[is_texts == 1], diameters[is_texts == 1], hw_ratios[is_texts == 1], s=4)
        ax.set_xlabel('swt')
        ax.set_ylabel('diameter')
        ax.set_zlabel('h/w ratio')
        plt.show()

    @staticmethod
    def draw_rect(img_path, dataset_path):
        data = load_dataset(dataset_path)
        src = cv2.imread(img_path)

        widths, heights, topleft_pts = [], [], []
        for datum in list(filter(lambda x: x['is_text'] == 0, data)):
            topleft_pts.append((datum['topleft_pt']['y'], datum['topleft_pt']['x']))
            widths.append(datum['width'])
            heights.append(datum['height'])

        label(src, topleft_pts, heights, widths, (0, 0, 255))

        widths, heights, topleft_pts = [], [], []
        for datum in list(filter(lambda x: x['is_text'] == 1, data)):
            topleft_pts.append((datum['topleft_pt']['y'], datum['topleft_pt']['x']))
            widths.append(datum['width'])
            heights.append(datum['height'])

        label(src, topleft_pts, heights, widths, (255, 0, 0))

    def svm(self):
        data1 = load_dataset('../output/006_words.json')
        data2 = load_dataset('../output/007_words.json')
        data3 = load_dataset('../output/008_words.json')
        data4 = load_dataset('../output/009_words.json')
        data5 = load_dataset('../output/010_words.json')

        print('count data: {}'.format(len(data1) + len(data2) + len(data3) + len(data4)))

        self.logger.info('Data is preparing...')
        x1, y1 = [], []
        x2, y2 = [], []
        x3, y3 = [], []
        x4, y4 = [], []

        x_test, y_test = [], []
        for datum in tqdm(data1):
            diameter = math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height'])
            feature = datum['percent_hist'] + [datum['height'] / datum['width'], datum['swt'], diameter]
            x1.append(feature)
            y1.append(datum['is_text'])

        for datum in tqdm(data2):
            diameter = math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height'])
            feature = datum['percent_hist'] + [datum['height'] / datum['width'], datum['swt'], diameter]
            x2.append(feature)
            y2.append(datum['is_text'])

        for datum in tqdm(data3):
            diameter = math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height'])
            feature = datum['percent_hist'] + [datum['height'] / datum['width'], datum['swt'], diameter]
            x3.append(feature)
            y3.append(datum['is_text'])

        for datum in tqdm(data4):
            diameter = math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height'])
            feature = datum['percent_hist'] + [datum['height'] / datum['width'], datum['swt'], diameter]
            x4.append(feature)
            y4.append(datum['is_text'])

        len_y_true_1 = sum(y1)
        x1, y1 = x1[:len_y_true_1 * 2], y1[:len_y_true_1 * 2]

        len_y_true_2 = sum(y2)
        x2, y2 = x2[:len_y_true_2 * 2], y2[:len_y_true_2 * 2]

        len_y_true_3 = sum(y3)
        x3, y3 = x3[:len_y_true_3 * 2], y3[:len_y_true_3 * 2]

        len_y_true_4 = sum(y4)
        x4, y4 = x4[:len_y_true_4 * 2], y4[:len_y_true_4 * 2]

        x = x1 + x2 + x3 + x4
        y = y1 + y2 + y3 + y4

        for datum in tqdm(data5):
            diameter = math.sqrt(datum['width'] * datum['width'] + datum['height'] * datum['height'])
            feature = datum['percent_hist'] + [datum['height'] / datum['width'], datum['swt'], diameter]
            x_test.append(feature)
            y_test.append(datum['is_text'])

        train(x, y, x_test, y_test)


if __name__ == '__main__':
    fire.Fire(Main)
