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
        '''Show some basic information and network structure of the NeNe.
        '''

        print(
            '---------------NeNe neural network model summary------------------'
        )
        print('Basic Information:')
        print('Depth:%d\nInput shape:%d\nOutput shape:%d' %
              (self.net_depth, self.input_num, self.output_num))
        print('Network structure:')
        print('%+15s%+15s%+15s%+15s' % ('Name', 'Neural_Num', 'Param_Num',
                                        'Activation'))
        print(
            '------------------------------------------------------------------'
        )
        print('%+15s%+15s%+15s%+15s\n' % ('Layer0(Input)', self.input_num, 0,
                                          self.layer_sequence[0].acti))
        sumParam, sumNeural = 0, 0
        for i in range(1, len(self.layer_sequence)):
            param = (self.layer_sequence[i - 1].neural_num + 1
                     ) * self.layer_sequence[i].neural_num
            sumParam += param
            sumNeural += self.layer_sequence[i].neural_num
            print('%+15s%+15s%+15s%+15s\n' %
                  ('Layer' + str(i), self.layer_sequence[i].neural_num, param,
                   self.layer_sequence[i].acti))
        print('%+15s%+15s%+15s%+15s' % ('Summary', sumNeural, sumParam,
                                        '------'))
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

    def backwardUpdate(self, y, networkError, lr):
        #assume that networkError has be calculated.
        d_thislayer = networkError
        for i in reversed(range(1, len(self.layer_sequence))):
            i_1 = i - 1
            # update bias
            d_thislayer = self.layer_sequence[i].backwardPropagation(
                y[i], d_thislayer, lr=lr)
            # update weight (backup weight first)
            d_nextlayer = d_thislayer * np.transpose(self.layer_sequence[i_1])
            self.layer_sequence[i_1] = self.layer_sequence - lr * np.transpose(
                self.layer_sequence[i_1]) * d_thislayer
            d_thislayer = d_nextlayer
        return
    
    def inputVaildCheck(self,input_data):
        x_,y_=input_data
        x_,y_=np.array(x_),np.array(y_)
        (inshape_x,inshape_y),(outshape_x,outshape_y)=x_.shape,y_.shape
        assert inshape_x==outshape_x
        assert inshape_y==self.input_num
        assert outshape_y==self.output_num
        return x_,y_

    def fit(self, train_data, valid_data=None, epoch=1, lr=0.001, batch_size=1):
        # validation check
        assert isinstance(train_data,(list,tuple,np.array))
        if not valid_data is None:
            assert isinstance(valid_data,(list,tuple,np.array))
        assert isinstance(epoch,int)
        assert isinstance(lr,float)
        assert isinstance(batch_size,int)
        # data
        x_train,y_train=self.inputVaildCheck(train_data)
        if valid_data:

        return

    def predict(self, x_test, y_test=None):
        return
