import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import os
import cv2
import math
from tqdm import tqdm

kanji_sliced_dir = '../output/japanese_letter/resized/kanji/'
hika_sliced_dir = '../output/japanese_letter/resized/hira_kata/'
false_sliced_dir = '../output/negative_slice/resized/'
test_slice_dir = '../example/resized/'

true_features, false_features = [], []

for slice_img in tqdm([i.split('.')[0] for i in os.listdir(kanji_sliced_dir)]):
    image = cv2.imread(kanji_sliced_dir + slice_img + '.jpg', 0)
    fd = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(4, 4), block_norm='L2')
    true_features.append(fd)

for slice_img in tqdm([i.split('.')[0] for i in os.listdir(hika_sliced_dir)]):
    image = cv2.imread(hika_sliced_dir + slice_img + '.jpg', 0)
    fd = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(4, 4), block_norm='L2')
    true_features.append(fd)

false_data = [i.split('.')[0] for i in os.listdir(false_sliced_dir)]
small_false_data = false_data[:len(true_features)]
for slice_img in tqdm(small_false_data):
    image = cv2.imread(false_sliced_dir + slice_img + '.jpg', 0)
    fd = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(4, 4), block_norm='L2')
    false_features.append(fd)

y_train = ([1] * len(true_features)) + ([0] * len(false_features))
x_train = true_features + false_features

clf = svm.SVC()
clf.fit(x_train, y_train)

test_features = []
test_filename = []
for slice_img in tqdm([i.split('.')[0] for i in os.listdir(test_slice_dir)]):
    image = cv2.imread(test_slice_dir + slice_img + '.jpg', 0)
    fd = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(4, 4), block_norm='L2')
    test_features.append(fd)
    test_filename.append(slice_img)

y_predicted = clf.predict(test_features)

for idx in range(len(y_predicted)):
    if y_predicted[idx]:
        print(test_filename[idx])
