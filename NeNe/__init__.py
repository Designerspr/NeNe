'''Main class of NeNe.
'''

from .Activation import *
from .Layer import *
import numpy as np


class NeNe(object):
    def __init__(self):
        # Main model list
        self.layer_sequence = list()
        self.weight_sequence = list()
        self.input_num = None
        self.output_num = None
        self.net_depth = 0

    def add(self, add_layer):
        # first layer must be InputLayer
        if len(self.layer_sequence) == 0:
            assert isinstance(add_layer, InputLayer)
        else:
            assert isinstance(add_layer, Layer)
        # add the layer information into the class
        # weight matrix created if it's hidden layer.
        # default to give the matrix with random values.
        self.layer_sequence.append(add_layer)
        if self.net_depth != 0:
            n, m = self.output_num, add_layer.neural_num
            weight_matrix = np.random.randn(n, m)
            self.weight_sequence.append(weight_matrix)
        else:
            self.input_num = add_layer.neural_num
        self.net_depth += 1
        self.output_num = add_layer.neural_num

    def summary(self):
        print('-----NeNe neural network model summary---')
        print('-----------------------------------------')
        print('No.             Num             Param_Num       Activation')
        print('Layer0       ',self.input_num,'      ',self.input_num)
        sumParam=self.input_num
        for i in range(1,len(self.layer_sequence)):
            param=(self.layer_sequence[i-1].neural_num+1)*self.layer_sequence[i].neural_num
            sumParam +=param
            print('Layer',i,':      ',self.layer_sequence[i].neural_num,'      ',param,'        ',self.layer_sequence[i].acti)
        print('Summary      ',self.net_depth,'      ',sumParam)
        return

    def forwardCalculation(self, x):
        '''calculating forward calculation with given data.
      
        Arguments:
            x {ndarray} -- input data
        
        Returns:
            [ndarray] -- output data
        '''
        output = x
        networkValue = list([output])
        for (layer_index, weight_matrix) in enumerate(self.weight_sequence):
            # layer connection
            output = output * weight_matrix
            # inner layer
            output = self.layer_sequence[layer_index
                                         + 1].forwardCalculation(output)
            networkValue.append([output])

        return output, networkValue

    def backwardUpdate(self,y,networkError,lr):
        d_thislayer=0
        return

    def fit(self, x_train, y_train, x_valid, y_valid, lr=0.001, epoch=1):
        '''main function,for NeNe to use train data to create 
        
        Arguments:
            x_train {[type]} -- [description]
            y_train {[type]} -- [description]
            x_valid {[type]} -- [description]
            y_valid {[type]} -- [description]
        
        Keyword Arguments:
            lr {float} -- [description] (default: {0.001})
            epoch {int} -- [description] (default: {1})
        '''

        return

    def predict(self, x_test, y_test=None):
        return
