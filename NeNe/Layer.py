'''
Libary about layers.
'''

import numpy as np
import random
from .Activation import *


class Layer(object):
    '''the normal layer class of NeNe. Providing 1-d layer.
    To get start, using following param:
    Arguments:
        num {int} -- the number of neural used in this layer.
        
    Keyword Arguments:
        activation {NeNe.Activation.Activation} -- the activation used in this layer. Default to be linear
        init_seed {str} -- the way to initialize the bias: set-zero('zero') or in gaussian('norm'). (default: {'norm'})
    '''

    def __init__(self, num, activation=None, init_seed='norm'):
        # activation
        if not activation:
            self.activation = Linear()
        else:
            assert isinstance(activation, Base_Activation)
            self.activation = activation
        # bias
        assert init_seed in ('zero', 'norm')
        assert isinstance(num, int)
        self.num = num
        if init_seed == 'zero':
            self.bias = np.zeros(num)
        else:
            self.bias = np.random.randn(num)
        return

    @property
    def neural_num(self):
        return self.num

    @property
    def acti(self):
        return self.activation.name

    def forwardPropagation(self, value_input):
        '''forward propagation with this layer only.
        '''
        value_output = value_input + self.bias
        value_output = self.activation.forward(value_output)
        return value_output

    def backwardPropagation(self, value_output, err_output, lr):
        '''backward propagation in this layer, passing values to the last layer.
        '''
        # activation function
        delta_bias = self.activation.get_derivative(value_output) * err_output
        # update bias
        self.bias = self.bias - delta_bias * lr
        # pass the value to train the weight and the
        return delta_bias


class InputLayer(Layer):
    '''the input layer class of NeNe. To be the input layer,there is no bias or activation.
    You have to set this to be the first layer of the NeNe, or class will raise exception.
    To get start, using following param:
    Arguments:
        num {int} -- the number of neural used in this layer.
        
    Keyword Arguments:
        activation {NeNe.Activation.Activation} -- the activation used in this layer. Default to be linear
        init_seed {str} -- the way to initialize the bias: set-zero('zero') or in gaussian('norm'). (default: {'norm'})
    '''

    def __init__(self, num):
        # no activation
        self.activation = Linear()
        # no bias
        assert isinstance(num, int)
        self.num = num
        self.bias = np.zeros(num)
        return