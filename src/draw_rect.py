import cv2
import fire

from modules.file_manager import load_dataset
from modules.label_rect import label


def draw_rect(id, dataset_path):
    data = load_dataset(dataset_path)
    src = cv2.imread('../../danbooru/resized/images/%s.jpg' % id)

    widths, heights, topleft_pts = [], [], []
    for datum in list(filter(lambda x: x['is_text'] in [-1, 1, 0], data)):
        topleft_pts.append((datum['topleft_pt']['y'], datum['topleft_pt']['x']))
        widths.append(datum['width'])
        heights.append(datum['height'])
    label(src, topleft_pts, heights, widths, (255, 0, 0))


if __name__ == '__main__':
    fire.Fire(draw_rect)
