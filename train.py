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

y_shifted, L01, L10 = label_shiftting(y.copy(), p = 0.2)
y_shifted = to_categorical(y_shifted, 2)

X_test = np.array(X_test)
y_test = np.array(y_test)


#---Train---
from tensorflow.compat.v1.keras.applications import ResNet101

res = ResNet101(weights='imagenet', include_top = False, classes = 2, input_shape=(32,32,3))


model = Sequential()
model.add(res)
# model1.add(GlobalAveragePooling2D(name='GAP'))
model.add(Flatten(name='FC'))
# 
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax', name='dense'))
res.trainable = False
model.summary()

lr = 1e-3
sgd = SGD(lr = lr, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['acc'])

hist = model.fit(x=X, y=y_shifted, batch_size=32, epochs=32, verbose=1, callbacks=None, validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

plt.figure(figsize=(18,8))

plt.suptitle('Loss and Accuracy Plots', fontsize=18)

plt.subplot(1,2,1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Number of epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt.subplot(1,2,2)
plt.plot(hist.history['acc'], label='Train Accuracy')
plt.plot(hist.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Number of epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.show()

y_pred = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred, axis = 1)
print(accuracy_score(y_test, y_pred))









