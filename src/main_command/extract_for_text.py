import logging
import math
from pathlib import Path
from multiprocessing import Process, Manager

import cv2
from tqdm import tqdm

# Modules
from text_detection import text_detection
from modules.file_manager import save_with_is_text_value
from modules.utils import histogram_calculate_parallel


def extract_for_text(path, labeled_file):
    image_file = Path(str(path))
    acceptable_types = ['.jpg', '.JPG', '.jpeg', '.JPEG']

    logging.info('Input path: %s', path)
    logging.info('Absolute path: %s', image_file.resolve())

    if not image_file.is_file() or image_file.suffix not in acceptable_types:
        logging.error('File is not in %s types.', acceptable_types)
        quit()

    src = cv2.imread(path)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    word_list = text_detection(src, src.shape[0])

    logging.info('The histogram is calculating...')

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
        is_texts.append(-1)
        hists.append(word['hist'])

    save_with_is_text_value(labeled_file, swts, heights, widths, topleft_pts, is_texts, hists)
