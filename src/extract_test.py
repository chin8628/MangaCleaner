import logging
import sys
import cv2
import fire
import os

from text_detection import text_detection
from modules.file_manager import save_by_dict

sys.setrecursionlimit(10000)


def testing_extract():
    image_dir = '../../Manga109-small/images/'
    output_dir = '../output/test/'
    image_dir_list = os.listdir(image_dir)

    for title in image_dir_list:
        img_test_dir = image_dir + title + '/test/'

        print(image_dir_list, title)

        if not os.path.exists(output_dir + title):
            os.makedirs(output_dir + title)

        for page_file in os.listdir(img_test_dir):
            page_id = int(page_file.split('.')[0])
            path = img_test_dir + page_file
            src = cv2.imread(path)

            index, data = 0, []
            for datum in text_detection(src):
                data.append({
                    'id': index,
                    'height': datum['height'],
                    'width': datum['width'],
                    'is_text': -1,
                    'topleft_pt': {'x': datum['x'], 'y': datum['y']}
                })

                index += 1

            print('SAVED! ' + output_dir + title + '/%03d.json' % page_id)
            save_by_dict(output_dir + title + '/%03d.json' % page_id, data)

    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)

    fire.Fire(testing_extract)
