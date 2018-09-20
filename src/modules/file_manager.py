import json
import logging
import cv2
from typing import Dict


def save(file_name: str, swts: list, heights: list, widths: list, topleft_pts: list, is_texts: list,
         hists: list):
    logger = logging.getLogger(__name__)
    data = []

    logger.info('Preparing data for saving...')
    for index in range(0, len(swts)):
        data.append({
            'id': index,
            'swt': float(swts[index]),
            'height': int(heights[index]),
            'width': int(widths[index]),
            'topleft_pt': {
                'x': int(topleft_pts[index][1]),
                'y': int(topleft_pts[index][0])
            },
            'is_text': 1 if is_texts[index] else 0,
            'hist': [float(i) for i in hists[index]]
        })

    logger.info('Saving...')
    with open(file_name, 'w') as fp:
        json.dump(data, fp)


def save_with_is_text_value(file_name: str, swts: list, heights: list, widths: list, topleft_pts: list, is_texts: list,
                            hists: list):
    logger = logging.getLogger(__name__)
    data = []

    logger.info('Preparing data for saving...')
    for index in range(0, len(swts)):
        data.append({
            'id': index,
            'swt': float(swts[index]),
            'height': int(heights[index]),
            'width': int(widths[index]),
            'topleft_pt': {
                'x': int(topleft_pts[index][1]),
                'y': int(topleft_pts[index][0])
            },
            'is_text': is_texts[index],
            'hist': [float(i) for i in hists[index]]
        })

    logger.info('Saving...')
    with open(file_name, 'w') as fp:
        json.dump(data, fp)


def save_by_dict(file_name: str, data):
    logger = logging.getLogger(__name__)
    logger.info('Saving... {}'.format(file_name))
    with open(file_name, 'w') as fp:
        json.dump(data, fp)


def load_dataset(file_name: str) -> dict:
    """
    :param file_name: data.json etc...
    :return: list(dict(swt, height, width, topleft_pt, is_text))
    """
    with open(file_name) as fp:
        data = json.load(fp)

    return data
