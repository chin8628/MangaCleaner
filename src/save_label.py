import logging
from pathlib import Path
import sys
import math

import cv2
import fire
import numpy as np

# Modules
from tqdm import tqdm

from text_detection import text_detection
from modules.danbooru import Danbooru
from modules.file_manager import save
from modules.utils import histogram_calculate

sys.setrecursionlimit(10000)


def save_label(id, output_path):
    annotation_path = '../../danbooru/annotations/%s.json' % id
    path = '../../danbooru/images/resized/%s.jpg' % id
    image_file = Path(str(path))

    logging.info('Absolute annotation path: %s', Path(annotation_path).resolve())
    logging.info('Absolute img path: %s', image_file.resolve())

    src = cv2.imread(path)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    swts, heights, widths, topleft_pts, is_texts, hists = [], [], [], [], [], []
    for text_area in tqdm(Danbooru(id).get_text_area()):
        x1, y1, h, w = text_area['x'], text_area['y'], text_area['height'], text_area['width']
        x2, y2 = x1 + w, y1 + h

        hist = histogram_calculate(src_gray[y1: y2+1, x1: x2+1])
        src[y1: y2+1, x1: x2+1] = 255

        swts.append(0)
        heights.append(y2 - y1)
        widths.append(x2 - x1)
        topleft_pts.append((y1, x1))
        is_texts.append(1)
        hists.append(hist)

    for word in text_detection(src):
        x1, y1 = word['topleft_pt'][1], word['topleft_pt'][0]
        x2, y2 = word['topleft_pt'][1] + word['width'], word['topleft_pt'][0] + word['height']
        hist = histogram_calculate(src_gray[y1: y2+1, x1: x2+1])

        swts.append(0)
        heights.append(word['height'])
        widths.append(word['width'])
        topleft_pts.append(word['topleft_pt'])
        is_texts.append(0)
        hists.append(hist)

    save(output_path, swts, heights, widths, topleft_pts, is_texts, hists)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)

    fire.Fire(save_label)
