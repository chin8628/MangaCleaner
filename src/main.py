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
from main_command.extract_for_text import extract_for_text as extract_for_text_command

sys.setrecursionlimit(10000)


class Main:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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

    def svm(self):
        train_command()

    def extract_for_test(self, path, output_file):
        extract_for_text_command(
            '../../Dataset_Manga/Manga109/images/' + path,
            output_file
        )


if __name__ == '__main__':
    fire.Fire(Main)
