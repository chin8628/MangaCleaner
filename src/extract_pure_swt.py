import cv2
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.manga109 import Manga109
from modules.file_manager import save_by_dict
from text_detection import text_detection
from tqdm import tqdm


def main():
    image_dir = '../../Manga109-small/images/'
    output_dir = '../output/pure_swt/'
    image_dir_list = os.listdir(image_dir)

    for title in image_dir_list:
        img_test_dir = image_dir + title + '/test/'

        print(image_dir_list, title)

        if not os.path.exists(output_dir + title):
            os.makedirs(output_dir + title)

        for page_file in tqdm(os.listdir(img_test_dir)):
            page_id = int(page_file.split('.')[0])
            path = img_test_dir + page_file
            src = cv2.imread(path)

            index, data = 0, []
            for datum in text_detection(src):
                data.append({
                    'id': index,
                    'height': int(datum['height']),
                    'width': int(datum['width']),
                    'is_text': 1,
                    'topleft_pt': {'x': int(datum['x']), 'y': int(datum['y'])}
                })

                index += 1

            # print('SAVED! ' + output_dir + title + '/%03d.json' % page_id)
            save_by_dict(output_dir + title + '/%03d.json' % page_id, data)


if __name__ == '__main__':
    main()
