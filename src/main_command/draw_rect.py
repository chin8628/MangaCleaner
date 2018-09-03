import cv2

from modules.file_manager import load_dataset
from modules.label_rect import label


def draw_rect(img_path, dataset_path):
    data = load_dataset(dataset_path)
    src = cv2.imread(img_path)

    # widths, heights, topleft_pts = [], [], []
    # for datum in list(filter(lambda x: x['is_text'] == 0, data)):
    #     topleft_pts.append(
    #         (datum['topleft_pt']['y'], datum['topleft_pt']['x']))
    #     widths.append(datum['width'])
    #     heights.append(datum['height'])
    # label(src, topleft_pts, heights, widths, (0, 0, 255))

    widths, heights, topleft_pts = [], [], []
    for datum in list(filter(lambda x: x['is_text'] == 1, data)):
        topleft_pts.append((datum['topleft_pt']['y'], datum['topleft_pt']['x']))
        widths.append(datum['width'])
        heights.append(datum['height'])
    label(src, topleft_pts, heights, widths, (255, 0, 0))

    # widths, heights, topleft_pts = [], [], []
    # for datum in list(filter(lambda x: x['is_text'] == -1, data)):
    #     topleft_pts.append((datum['topleft_pt']['y'], datum['topleft_pt']['x']))
    #     widths.append(datum['width'])
    #     heights.append(datum['height'])
    # label(src, topleft_pts, heights, widths, (255, 0, 0))
