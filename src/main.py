import logging
import math
from pathlib import Path
from multiprocessing import Process, Manager
import os
import sys

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
from modules.utils import histogram_calculate_parallel

from main_command.svm import train as train_command
from main_command.draw_rect import draw_rect as draw_rect_command
from main_command.extract_for_text import extract_for_text as extract_for_text_command

sys.setrecursionlimit(10000)


class Main:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_labeled_data(self, page, path, labeled_file, annotation_path):
        # expected_height = 1200

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
        hists, is_texts = [], []
        for path in paths:
            self.logger.info('%s is Loading...', path)
            data = load_dataset(path)

            for datum in data:
                hists.append(datum['hist'])
                is_texts.append(datum['is_text'])

        hists, is_texts = np.array(hists), np.array(is_texts)

        skew_not_text, skew_text = [], []

        for a in hists[is_texts == 0]:
            p = [i / (datum['width'] * datum['height']) for i in a]
            q = math.sqrt(sum([(i - 127.5)**2 * p[i] for i in range(0, 256)]))
            skew_not_text.append((q**(-3)) * sum([(i - 127.5)**3 * p[i] for i in range(0, 256)]))

        for a in hists[is_texts == 1]:
            p = [i / (datum['width'] * datum['height']) for i in a]
            q = math.sqrt(sum([(i - 127.5)**2 * p[i] for i in range(0, 256)]))
            skew_text.append((q**(-3)) * sum([(i - 127.5)**3 * p[i] for i in range(0, 256)]))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(
            skew_not_text,
            # [np.mean(i) for i in hists[is_texts == 0]],
            [np.std(i) for i in hists[is_texts == 0]],
            [sum(i[200:256]) / (datum['width'] * datum['height']) for i in hists[is_texts == 0]],
            s=4
        )
        ax.scatter(
            skew_text,
            # [np.mean(i) for i in hists[is_texts == 1]],
            [np.std(i) for i in hists[is_texts == 1]],
            [sum(i[200:256]) / (datum['width'] * datum['height']) for i in hists[is_texts == 1]],
            s=4
        )
        plt.show()

    @staticmethod
    def draw_rect(img_path, dataset_path):
        draw_rect_command('../../Dataset_Manga/Manga109/images/' + img_path, dataset_path)

    def svm(self):
        train_command()

    def extract_for_test(self):
        imagedir_dataset_path = '../../Dataset_Manga/Manga109/images/'
        test_images_path = open('./filename_test.txt', 'r').read().split('\n')
        test_json = ['.'.join(i.replace('-', '/').split('.')[0:-1]) for i in os.listdir('../output/')]
        failed_log = ['.'.join(i.replace('-', '/').split('.')[0:-1]) for i in os.listdir('../log/')]

        for path in test_images_path:
            if path in test_json:
                self.logger.info('Already extracted >> {}'.format(imagedir_dataset_path + path))
                continue
            elif path in failed_log:
                self.logger.info('Already failed >> {}'.format(imagedir_dataset_path + path))
                continue

            self.logger.info('extracting >> {}'.format(imagedir_dataset_path + path))
            try:
                extract_for_text_command(
                    imagedir_dataset_path + path,
                    '../output/' + path.replace('/', '-') + '.json'
                )
            except KeyboardInterrupt:
                file = open('../log/{}.txt'.format(path.replace('/', '-')), 'w')
                file.close()
                quit()
            except Exception as e:
                file = open('../log/{}.txt'.format(path.replace('/', '-')), 'w')
                file.write(str(e))
                self.logger.exception(e)
                file.close()


if __name__ == '__main__':
    fire.Fire(Main)
