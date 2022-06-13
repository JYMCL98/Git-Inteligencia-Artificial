# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:37:57 2022

@author: jymcl
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

X_train = np.loadtxt('D:/Documents/Data/Plants_train.csv', delimiter = ',')
Y_train = np.loadtxt('D:/Documents/Data/Plants_train_labels.csv', delimiter = ',')

X_test = np.loadtxt('D:/Documents/Data/Plants_test.csv', delimiter = ',')
Y_test = np.loadtxt('D:/Documents/Data/Plants_test_labels.csv', delimiter = ',')

X_train = X_train.reshape(len(X_train), 80, 80, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 80, 80, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train/255
X_test = X_test/255