import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread('./jp2.jpg', 0)
pure = cv2.imread('./jp2.jpg', 0)

start_x, start_y = 98, 99
x, y, window_size = start_x, start_y, 40
margin_y = 0
margin_x = 6

cnt = 0
chr_id = 0
cnt_y = 0

while y < 2365:
    cnt_x = 0

    while x < 2625:
        cv2.rectangle(
            img,
            (x, y),
            (x+window_size, y+window_size),
            (0, 0, 0),
            2
        )

        roi = pure[y: y+window_size+1, x: x+window_size+1]

        x += window_size + margin_x

        cv2.imwrite('../output/slice/%d.jpg' % chr_id, roi)
        chr_id += 1

        cnt_x += 1

        if cnt_x % 10 == 0:
            x += 53

    y += window_size + margin_y
    x = start_x

    cnt_y += 1

    if cnt_y == 10:
        y = 574
    elif cnt_y == 20:
        y = 1043
    elif cnt_y == 30:
        y = 1509
    elif cnt_y == 40:
        y = 1965

plt.imshow(img, cmap="gray")
plt.show()
