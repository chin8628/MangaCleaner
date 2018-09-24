import os
import cv2
import math
import json

input_dir = '../example/slice/'

for id_img in [i for i in os.listdir(input_dir)]:
    resized_img_path = '../example/resized/%s' % id_img
    img_path = input_dir + id_img

    print(img_path)

    src = cv2.imread(img_path)
    src_height, src_width = src.shape[0], src.shape[1]

    if src_height <= 20:
        cv2.imwrite(resized_img_path, src)
        continue

    ratio = src_height / 20
    new_h, new_w = math.floor(src_height / ratio), math.floor(src_width / ratio)

    src = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(resized_img_path, src)
