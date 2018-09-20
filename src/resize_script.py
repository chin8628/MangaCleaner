import os
import cv2
import math

for img_path in filter(lambda x: x != 'resized', os.listdir('../../danbooru/images/')):
    print(img_path)

    src = cv2.imread('../../danbooru/images/' + img_path)
    src_height, src_width = src.shape[0], src.shape[1]

    if src_height <= 1200:
        continue

    ratio = src_height / 1200
    new_h, new_w = math.floor(src_height / ratio), math.floor(src_width / ratio)

    src = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite('../../danbooru/images/resized/%s' % img_path, src)
