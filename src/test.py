import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

with open('./a.json') as fp:
    data = json.load(fp)

img = cv2.imread('./a.jpg')

for datum in data:
    x, y, w, h = datum['x'], datum['y'], datum['width'], datum['height']
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

plt.imshow(img)
plt.show()
