import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from modules.manga109_annotation import Manga109Annotation

page_id = 36
manga_title = 'arisa'

annotation_file = '../test_dataset/%s/%03d.json' % (manga_title, page_id)
image_file = '../test_dataset/%s/%03d.jpg' % (manga_title, page_id)

text_region = {}
with open(annotation_file, 'r') as fp:
    text_region = json.load(fp)

img = cv2.imread(image_file)

for id in text_region:
    topleft_pt, bottomright_pt = text_region[id]['topleft_pt'], text_region[id]['bottomright_pt']
    cv2.putText(
        img,
        id,
        (topleft_pt['x'], topleft_pt['y']),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (0, 0, 0)
    )
    cv2.rectangle(
        img,
        (topleft_pt['x'], topleft_pt['y']),
        (bottomright_pt['x'], bottomright_pt['y']),
        (255, 0, 0),
        2
    )

plt.imshow(img)
plt.show()
