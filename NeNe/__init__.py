'''Main class of NeNe.
'''

from Activation import *
from Layer import *
import numpy as np


class NeNe(object):
    def __init__(self):
        # Main model list
        self.sequence=list()
        self.output_shape=None
        self.net_depth=0
        self.num_parameters=0
        return

    def add(self,Layer):
        Layer
        return

    def summary(self):
        print('Model bulit via NeNe')
        print('--------------------')

        return

    def fit(self, x_train, y_train, x_valid, y_valid, lr=0.001, epoch=1):
        return

    def predict(self, x_test, y_test=None):
        return
