'''
Libary providing activation.
'''

import numpy as np

class Activation(object):
    '''
    The basic class of Activation.  
        If you want to create your own activate function ,the new class should inhertance from this. 
        Besides,you should rewrite get_output function and get_derivative function, which defines how the activation works.
    Activation equals to linear activation defaultly.
    Attention: Activation class works for 1-d network only.
    '''
    def __init__(self):
        # Nothing to be done.
        return

    def forward(self,x_1d):
        y_1d=np.zeros(x_1d.shape)
        for index in range(len(x_1d)):
            y_1d[index]=self.get_output(x_1d[index])
        return y_1d
    def derivative(self,y_1d):
        x_1d=np.zeros(y_1d.shape)
        for index in range(len(y_1d)):
            x_1d[index]=self.get_derivative(y_1d[index])
        return x_1d
    def get_output(self,x): 
        # Need to be rewrite
        return x
    
    def get_derivative(self,y):
        # Need to be rewrite
        return 1


class Linear(Activation):
    # Pre-defined activation: Linear
    def __init__(self):
        return
    def get_output(self,x):
        return x
    
    def get_derivative(self,y):
        return 1


class ReLu(Activation):
    # Pre-defined activation: ReLu
    def __init__(self):
        return 
    def get_output(self,x):
        if x>0:
            return x
        else:
            return 0
    
    def get_derivative(self,y):
        if y>0:
            return 1
        else:
            return 0


class Sigmond(Activation):
    # Pre-defined activation: Sigmond
    def  __init__(self):
        return
    def get_output(self,x):
        return 1/(1+np.exp(x))
    def get_derivative(self,y):
        return y(1-y)


class Tanh(Activation):
    # Pre-defined activation: Tanh
    def __init__(self):
        return
    def get_output(self,x):
        return np.tanh(x)
    def get_derivative(self,y):
        return 1-y**2


    