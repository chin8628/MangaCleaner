import os
import cv2
import math
import json

for img_name in os.listdir('../../danbooru/images/'):
    id_img = img_name.split('.')[0]
    img_path = '../../danbooru/images/' + img_name
    annotation_path = '../../danbooru/annotations/' + id_img + '.json'
    resized_img_path = '../../danbooru/resized/images/' + id_img + '.jpg'
    resized_annotation_path = '../../danbooru/resized/annotations/' + id_img + '.json'

    print(id_img)

    src = cv2.imread(img_path)
    src_height, src_width = src.shape[0], src.shape[1]

    with open(annotation_path, 'r') as fp:
        data = json.load(fp)

    if src_height <= 1200:
        cv2.imwrite(resized_img_path, src)
        with open(resized_annotation_path, 'w') as fp:
            json.dump(data, fp)

        continue

    ratio = src_height / 1200
    new_h, new_w = math.floor(src_height / ratio), math.floor(src_width / ratio)

    for datum in data:
        datum['x'] = math.floor(datum['x'] / ratio)
        datum['y'] = math.floor(datum['y'] / ratio)
        datum['height'] = math.floor(datum['height'] / ratio)
        datum['width'] = math.floor(datum['width'] / ratio)

    with open(resized_annotation_path, 'w') as fp:
        json.dump(data, fp)

    src = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(resized_img_path, src)
