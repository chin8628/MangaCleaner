import cv2
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.manga109 import Manga109
from text_detection import text_detection


def main():
    img_dir = '../../Manga109-small/images/'
    annotation_dir = '../../Manga109-small/annotations/'
    output_window_dir = '../output/connected_comp_window_manga109/'

    for title in os.listdir(img_dir):
        manga109 = Manga109(title)

        for page_file in os.listdir(img_dir + title):
            page_id = int(page_file.split('.')[0])
            print(page_id)

            src = cv2.imread(img_dir + title + '/' + page_file)
            src_gray = cv2.imread(img_dir + title + '/' + page_file, 0)
            annotation = manga109.get_text_area(page_id)

            if not os.path.exists(output_window_dir + title + '/%03d/true/' % page_id):
                os.makedirs(output_window_dir + title + '/%03d/true/' % page_id)

            if not os.path.exists(output_window_dir + title + '/%03d/false/' % page_id):
                os.makedirs(output_window_dir + title + '/%03d/false/' % page_id)

            mask = np.zeros(src_gray.shape, np.uint8)

            for datum in annotation:
                x1, y1, w, h = datum['x'], datum['y'], datum['width'], datum['height']
                x2, y2 = x1 + w, y1 + h
                mask[y1:y2, x1:x2] = 1

            for datum in text_detection(src):
                x1, y1, w, h = datum['x'], datum['y'], datum['width'], datum['height']
                x2, y2 = x1 + w, y1 + h

                roi_mask = mask[y1:y2, x1:x2]

                if x1 < 0 or y1 < 0:
                    continue

                if x2 > src.shape[1] or y2 > src.shape[0]:
                    continue

                if sum(sum(roi_mask)) > (w * h) / 2:
                    cv2.imwrite(output_window_dir + '%s/%03d/true/y_%d_x_%d.jpg' %
                                (title, page_id, y1, x1), src[y1:y2, x1:x2])
                else:
                    cv2.imwrite(output_window_dir + '%s/%03d/false/y_%d_x_%d.jpg' %
                                (title, page_id, y1, x1), src[y1:y2, x1:x2])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)

    main()
