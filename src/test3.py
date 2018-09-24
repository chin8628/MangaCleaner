import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

annotation_dir = '../../danbooru/resized/annotations/'
image_dir = '../../danbooru/resized/images/'
cleaned_image_dir = '../output/cleaned/'
train_dir = '../output/train/'
slice_dir = '../output/negative_slice/imgs-26x26/'

ids = [i.split('.')[0] for i in os.listdir(annotation_dir)]
cnt = 0

for file_id in ids:
    print('id: %s (%d / %d)' % (file_id, cnt, len(ids)))

    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)

    with open(train_dir + file_id + '.json', 'r') as fp:
        train_data = json.load(fp)

    img = cv2.imread(cleaned_image_dir + file_id + '.jpg', 0)
    img_height, img_width = img.shape[0], img.shape[1]
    window_size = 25
    sliced_mark = np.zeros(img.shape)
    step = window_size

    for datum in [i for i in train_data if i['is_text'] == 0]:
        x, y = datum['topleft_pt']['x'], datum['topleft_pt']['y']
        w, h = datum['width'], datum['height']

        sliced_mark[y: y+h+1, x:x+w+1] = 1

    for y in range(0, img_height, step):
        if y > img_height:
            break

        for x in range(0, img_width, step):
            if sliced_mark[y, x] == 0 or x > img_width:
                continue

            if roi.shape[0] < window_size or roi.shape[1] < window_size:
                continue

            cv2.imwrite('%s/id_%s_y_%d_x_%d.jpg' % (slice_dir, file_id, y, x), roi)

    cnt += 1
