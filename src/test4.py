import cv2

img = cv2.imread('../example/text.jpg', 0)
h, w = img.shape
window_size = 25
step = 5

for y in range(0, h, step):
    for x in range(0, w, step):
        roi = img[y:y+window_size, x:x+window_size]

        if roi.shape[0] < window_size or roi.shape[1] < window_size:
            continue

        cv2.imwrite('../example/slice/%d_%d.jpg' % (y, x), roi)
