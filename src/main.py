import logging
import math
from pathlib import Path

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Modules
from text_detection import text_detection
from modules.manga109_annotation import Manga109Annotation
from modules.file_manager import save, load


class Main:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_labeled_data(self, path, labeld_file):
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

        # if src.shape[0] > int(expected_height):
        #     src = cv2.resize(
        #         cv2.imread(path),
        #         None,
        #         fx=expected_height / src.shape[0],
        #         fy=expected_height / src.shape[0]
        #     )

        manga109 = Manga109Annotation(annotation_path, 6)
        manga109_text_area_list = manga109.get_text_area_list()

        swt_values_list = []
        heights_list, widths_list, diameters_list, hw_ratio_list = [], [], [], []
        topleft_pts_list = []
        is_text_list = []

        margin = 5
        for text_area in manga109_text_area_list:

            topleft_pt, bottomright_pt = text_area[0], text_area[1]
            roi = src[
                  topleft_pt[0] - margin:bottomright_pt[0] + margin,
                  topleft_pt[1] - margin:bottomright_pt[1] + margin
                  ]

            data = text_detection(roi, roi.shape[0])
            swt_values, heights, widths, diameters, topleft_pts, letter_images, hw_ratio = data

            for index in range(0, len(swt_values)):
                swt_values_list.append(swt_values[index])
                heights_list.append(heights[index])
                widths_list.append(widths[index])
                diameters_list.append(diameters[index])
                topleft_pts_list.append(
                    (
                        topleft_pts[index][0] + topleft_pt[0] - margin,
                        topleft_pts[index][1] + topleft_pt[1] - margin
                    )
                )
                is_text_list.append(True)
                hw_ratio_list.append(hw_ratio[index])

        data = text_detection(src, src.shape[0])
        swt_values, heights, widths, diameters, topleft_pts, letter_images, hw_ratio = data

        for index in range(0, len(swt_values)):
            is_existed = False

            for index_2 in range(0, len(swt_values_list)):
                if swt_values_list[index_2] == swt_values[index] and heights_list[index_2] == heights[index] and \
                        widths_list[index_2] == widths[index] and topleft_pts_list[index_2] == topleft_pts[index]:
                    is_existed = True
                    break

            if is_existed:
                continue

            swt_values_list.append(swt_values[index])
            heights_list.append(heights[index])
            widths_list.append(widths[index])
            diameters_list.append(diameters[index])
            topleft_pts_list.append(topleft_pts[index])
            is_text_list.append(False)
            hw_ratio_list.append(hw_ratio[index])

        save(labeld_file, swt_values_list, heights_list, widths_list, topleft_pts_list, is_text_list)

    def plot_from_files(self, *paths):
        swts, widths, heights, diameters, hw_ratios, is_texts = [], [], [], [], [], []

        for path in paths:
            self.logger.info('%s is Loading...', path)
            data = load(path)

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
        ax.scatter(swts[is_texts == False], diameters[is_texts == False], hw_ratios[is_texts == False], s=4)
        ax.scatter(swts[is_texts == True], diameters[is_texts == True], hw_ratios[is_texts == True], s=4)
        ax.set_xlabel('swt')
        ax.set_ylabel('diameter')
        ax.set_zlabel('h/w ratio')
        plt.show()


if __name__ == '__main__':
    fire.Fire(Main)
