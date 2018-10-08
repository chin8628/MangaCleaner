import numpy as np
import json
from skimage.feature import hog
from sklearn import svm
import os
import cv2
import math
from tqdm import tqdm
from random import shuffle
from sklearn.externals import joblib
import matplotlib.pyplot as plt

dataset_dir = '../output/connected_comp_window_manga109/'

y_train, x_train = [], []
true_feature, false_feature = [], []

for title in os.listdir(dataset_dir):
    title_true_features, title_false_features = [], []

    for page in tqdm(os.listdir(dataset_dir + title)):
        for slice_img in os.listdir(dataset_dir + title + '/' + page + '/true/'):
            # print(dataset_dir + title + '/' + page + '/true/' + slice_img)

            image = cv2.imread(dataset_dir + title + '/' + page + '/true/' + slice_img, 0)
            h, w = image.shape
            fd = hog(image, pixels_per_cell=(h/10, w/10), cells_per_block=(2, 2), block_norm='L2')

            if len(fd) != 2916:
                continue

            title_true_features.append(fd)

        for slice_img in os.listdir(dataset_dir + title + '/' + page + '/false/'):
            image = cv2.imread(dataset_dir + title + '/' + page + '/false/' + slice_img, 0)
            h, w = image.shape
            fd = hog(image, pixels_per_cell=(h/10, w/10), cells_per_block=(2, 2), block_norm='L2')

            if len(fd) != 2916:
                continue

            title_false_features.append(fd)

    title_false_features = title_false_features[:len(title_true_features)]

    true_feature += title_true_features
    false_feature += title_false_features

x_train = true_feature + false_feature
y_train = ([1] * len(true_feature)) + ([0] * len(false_feature))

print('no. true: %d, no. false: %d' % (len(true_feature), len(false_feature)))
print('no. total feature: %d' % len(x_train))

clf = svm.SVC(cache_size=5000)
clf.fit(x_train, y_train)

joblib.dump(clf, 'model.pkl')

with open('trained_img_name.json', 'w') as fp:
    json.dump(os.listdir(dataset_dir)[:128], fp)
