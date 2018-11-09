import numpy as np
from random import random
import NeNe
from NeNe.Activation import *
from NeNe.Layer import *

def load_data(path):
    return x, y


def data_split(x, y, split_rate=0.1):
    return x1, y1, x2, y2


def data_shuffle(x, y):
    return x, y


def main():
    # model building
    model = NeNe.NeNe()
    model.add(num=2, activation='linear', init_seed='n_norm')
    model.add(num=10, activation='sigmoid', init_seed='n_norm')
    model.add(num=10, activation='sigmoid', init_seed='n_norm')
    model.add(num=2, activation='softmax', init_seed='n_norm')
    model.summary()
    # data preprocessing
    x, y = load_data('PATH')
    x, y = data_shuffle(x, y)
    x_train, y_train, x_test, y_test = data_split(x, y, split_rate=0.1)
    x_train, y_train, x_valid, y_valid = data_split(
        x_train, y_train, split_rate=0.1)

    model.fit(x_train, y_train, x_valid, y_valid, lr=0.001, epoch=100)
    model.predict(x_test, y_test)
