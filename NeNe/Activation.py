'''
Libary providing activation.
'''

import numpy as np


class Base_Activation(object):
    '''
    The basic class of Activation.  
        If you want to create your own activate function ,the new class should inhertance from this. 
        Besides,you should rewrite get_output function and get_derivative function, which defines how the activation works.
    Activation equals to linear activation defaultly.
    Attention: Activation class works for n*1-d network input data only.
    '''

    def __init__(self):
        # Nothing to be done.
        return

    @property
    def name(self):
        return 'linear'

    def forward(self, x):
        y = np.zeros(x.shape)
        for id_num in range(len(x)):
            y[id_num] = self.get_output(x[id_num])
        return y

    def derivative(self, y):
        x = np.zeros(y.shape)
        for id_num in range(len(y)):
            x[id_num] = self.get_output(y[id_num])
        return x

    def get_output(self, x):
        # Need to be rewrite
        return x

    def get_derivative(self, y):
        # Need to be rewrite
        deri=np.ones(y.shape)
        return deri


class Linear(Base_Activation):
    '''Pre-defined activation: Linear
    '''

    def __init__(self):
        return

    @property
    def name(self):
        return 'linear'

    def get_output(self, x):
        # Need to be rewrite
        return x

    def get_derivative(self, y):
        # Need to be rewrite
        deri=np.ones(y.shape)
        return deri


class ReLu(Base_Activation):
    '''Pre-defined activation: ReLu  
    ReLu is an activation perform like this:  
    f(x) = 0 (when x<=0); x (x>0)
    ''' 
    def __init__(self):
        return

    @property
    def name(self):
        return 'relu'

    def get_output(self, x):
        symbol=np.array(x>0,dtype=np.int)
        return symbol*x

    def get_derivative(self, y):
        symbol=np.array(y>0,dtype=np.int)
        return symbol


class Sigmond(Base_Activation):
    '''Pre-defined activation: Sigmond'''
    def __init__(self):
        return

    @property
    def name(self):
        return 'sigmond'

    def get_output(self, x):
        return 1 / (1 + np.exp(x))

    def get_derivative(self, y):
        return y*(1 - y)


class Tanh(Base_Activation):
    '''Pre-defined activation: Tanh'''
    def __init__(self):
        return

    @property
    def name(self):
        return 'tanh'

    def get_output(self, x):
        return np.tanh(x)

    def get_derivative(self, y):
        return 1 - y**2


class SoftMax(Base_Activation):
    '''Pre-defined activation: Softmax'''
    def __init__(self):
        return

    @property
    def name(self):
        return 'SoftMax'

    def get_output(self, x):
        x=x-np.max(x)
        x_softmax=np.exp(x)/np.sum(np.exp(x))
        return x_softmax

    def get_derivative(self, y):
        return y*(1-y)