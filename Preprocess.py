import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python import keras


import keras.backend as K
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.compat.v1.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.utils import Sequence
# from tensorflow.python.keras.utils.data_utils import Sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.utils import to_categorical

import albumentations as albu
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

np.random.seed(0)

(X_c, y_c), (X_test_c, y_test_c) = cifar100.load_data()

fine_label_list =  ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
                    'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 
                    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 
                    'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
                    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
                    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
                    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
                    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def label_shiftting(y, p = 0.2):
    L0 = np.where(y == 0)[0]
    L1 = np.where(y == 1)[0]
    A0 = np.random.choice(500, size=int(500*p), replace=False)
    A1 = np.random.choice(500, size=int(500*p), replace=False)
    y[L0[A0]] = 1
    y[L1[A1]] = 0
    return y, L0[A0], L1[A1]

X = []
y = []

for i,x in enumerate(X_c):
    if fine_label_list[y_c[i][0]] == 'caterpillar':
        X.append(x)
        y.append(0)
    elif fine_label_list[y_c[i][0]] == 'worm':
        X.append(x)
        y.append(1)
        
X_test = []
y_test = []
for i,x in enumerate(X_test_c):
    if fine_label_list[y_test_c[i][0]] == 'caterpillar':
        X_test.append(x)
        y_test.append(0)
    elif fine_label_list[y_test_c[i][0]] == 'worm':
        X_test.append(x)
        y_test.append(1)

X = np.array(X)
y = np.array(y)

y_shifted, L01, L10 = label_shiftting(y.copy(), p = 0.2) # 隨機打亂p部分的資料 
y_shifted = to_categorical(y_shifted, 2)

X_test = np.array(X_test)
y_test = np.array(y_test)