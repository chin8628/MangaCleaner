import cv2
import numpy as np
from matplotlib import pyplot as plt

def showImg(title, img, flag=0):
    f, ax = plt.subplots()
    f.canvas.set_window_title(title)

    if flag == 0:
        ax.imshow(img, interpolation = 'none')
    elif flag == 1:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation = 'none')
    elif flag == 2:
        ax.imshow(img, cmap='gray', interpolation = 'none')

img_src = cv2.imread('test2.jpg')

th, im_th = cv2.threshold(img_src, 220, 255, cv2.THRESH_BINARY_INV);
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_th, mask, (50,50), 255, None, None, cv2.FLOODFILL_MASK_ONLY)
mask = mask[1:-1,1:-1]

im2, contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
img_out = img_src.copy()
cv2.drawContours(img_out, contours, -1, (255,255,255), cv2.FILLED)

showImg("Src", img_src, 2)
showImg("Test", img_out, 2)
plt.show()
