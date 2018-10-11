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
import multiprocessing
import logging


def train(c_param=None, gamma_param=None):
    dataset_dir = '../output/connected_comp_window_manga109/'

    y_train, x_train = [], []
    true_feature, false_feature = [], []

    for title in os.listdir(dataset_dir):
        title_true_features, title_false_features = [], []

        for page in tqdm(os.listdir(dataset_dir + title)):
            for slice_img in os.listdir(dataset_dir + title + '/' + page + '/true/'):
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

    if c_param is None and gamma_param is None:
        clf = svm.SVC(cache_size=5000)
        clf.fit(x_train, y_train)
        model_name = 'model'
    else:
        clf = svm.SVC(cache_size=5000, C=2**c_param, gamma=2**gamma_param)
        clf.fit(x_train, y_train)
        model_name = 'c_%d_g_%d' % (c_param * 100, gamma_param * 100)

    joblib.dump(clf, './model/%s.pkl' % model_name)

    print('saved', model_name)

    return model_name


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__)

    train()
