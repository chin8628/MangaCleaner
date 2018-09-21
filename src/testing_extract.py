import logging
from pathlib import Path
import sys
import math
import itertools
from multiprocessing import Process, Manager

import cv2
import fire
import numpy as np

# Modules
from tqdm import tqdm

from text_detection import text_detection
from modules.danbooru import Danbooru
from modules.file_manager import save_by_dict
from modules.utils import histogram_calculate_parallel

sys.setrecursionlimit(10000)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def testing_extract(id, output_path):
    path = '../../danbooru/resized/images/%s.jpg' % id
    image_file = Path(str(path))
    logging.info('Absolute img path: %s', image_file.resolve())

    src = cv2.imread(path)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hists = []

    words = text_detection(src)

    processes = []
    hist_block = []
    process_number = 4
    wors_chunks = chunks(words, math.ceil(len(words) / process_number))
    with Manager() as manager:
        for word_chunk in wors_chunks:
            hist = manager.list()
            process = Process(target=histogram_calculate_parallel, args=(src_gray, word_chunk, hist))
            process.start()
            processes.append(process)
            hist_block.append(hist)

        for process in processes:
            process.join()

        hists = list(itertools.chain.from_iterable(hist_block))

    data = []
    for index in range(len(words)):
        data.append({
            'id': index,
            'swt': float(words[index]['swt']),
            'height': int(words[index]['height']),
            'width': int(words[index]['width']),
            'topleft_pt': {
                'x': int(words[index]['topleft_pt'][1]),
                'y': int(words[index]['topleft_pt'][0])
            },
            'is_text': -1,
            'hist': [float(i) for i in hists[index]]
        })

    save_by_dict(output_path, data)

    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)

    fire.Fire(testing_extract)
