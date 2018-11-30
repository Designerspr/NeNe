'''Main class of NeNe.
'''

from .Activation import *
from .Layer import *
from .Loss import *
from .LRD import *

import numpy as np
from progressbar import ProgressBar, Timer, Bar, Percentage


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
            [ndarray],[list] -- output data and the value of the network
        '''
        output = x
        networkValue = list([output])
        for (layer_index, weight_matrix) in enumerate(self.weight_sequence):
            # layer connection
            output = np.dot(output, weight_matrix)
            # inner layer
            output = self.layer_sequence[layer_index
                                         + 1].forwardPropagation(output)
            networkValue.append(output)
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
            d_nextlayer = np.dot(d_thislayer,
                                 np.transpose(self.weight_sequence[i_1]))
            self.weight_sequence[
                i_1] = self.weight_sequence[i_1] - lr * np.transpose(
                    y[i_1]) * d_thislayer
            d_thislayer = d_nextlayer

    def inputVaildCheck(self, input_data):
        '''Used to check if input data fits the requirement of the model
        Arguments:
            input_data {tuple} -- [description]
        
        Returns:
            [type] -- [description]
        '''

        x_, y_ = input_data
        x_, y_ = np.array(x_), np.array(y_)
        (inshape_x, inshape_y), (outshape_x, outshape_y) = x_.shape, y_.shape
        assert inshape_x == outshape_x
        assert inshape_y == self.input_num
        assert outshape_y == self.output_num
        return x_, y_

    def fit(self,
            train_data,
            valid_data=None,
            epoch=1,
            lrf=LRC(lr=0.01),
            batch_size=1,
            loss=None,
            accu_echo=False):
        # validation check
        assert isinstance(train_data, (list, tuple, np.array))
        if not valid_data is None:
            assert isinstance(valid_data, (list, tuple, np.array))
        assert isinstance(epoch, int)
        assert isinstance(lrf, LRC)
        assert isinstance(batch_size, int)
        if not loss is None:
            assert isinstance(loss, (CEL, MSE))
        else:
            # use MSE as default
            loss = MSE()

        # data split and transform
        x_train, y_train = self.inputVaildCheck(train_data)
        if not valid_data is None:
            x_valid, y_valid = self.inputVaildCheck(valid_data)
        # train
        batch_num = len(x_train)
        generator = data_generator(x_train, y_train, batch_size=batch_size)
        err, accu = 0, 0
        if accu_echo:
            widgets = [
                'Progress: ',
                Bar('■'), ' ',
                Percentage(), ' ',
                Timer(),
                ' loss=%.2f' % err,
                '  accu=%.2f%%' % accu
            ]
        else:
            widgets = [
                'Progress: ',
                Bar('■'), ' ',
                Percentage(), ' ',
                Timer(),
                ' loss=%.2f' % err
            ]
        pbar = ProgressBar(widgets=widgets, maxval=batch_num)

        for epoch_now in range(1, epoch + 1):
            print('Epoch %d/%d:' % (epoch_now, epoch))
            pbar.start()
            # train every batch
            for batch_id in range(batch_num):
                x_input, y_target = generator.__next__()

                # forward propagation
                y_output, network = self.forwardCalculation(x_input)

                # calculate error and print
                if accu_echo:
                    err, accu = loss.get_loss(
                        y_output, y_target, return_accu=True)
                else:
                    err = loss.get_loss(y_output, y_target, return_accu=False)
                lr=lrf.update(epoch_now,err)
                # update network
                y_error = loss.get_loss_deriv(y_output, y_target)
                self.backwardUpdate(network, y_error, lr=lr)
                pbar.update(value=batch_id+1)
            pbar.finish()
            # calculate accu,loss on the validation data if needed
            if not valid_data is None:
                y_predict = self.predict(x_valid)
                if accu_echo:
                    loss_this_epoch, accu_this_epoch = loss.get_loss(
                        y_predict, y_valid, return_accu=True)
                    print('valid_loss=%.4f;valid_accu=%.2f%%' %
                          (loss_this_epoch, accu_this_epoch))
                else:
                    loss_this_epoch = loss.get_loss(
                        y_predict, y_valid, return_accu=False)
                    print('valid_loss=%.4f' % (loss_this_epoch))

    def predict(self, x_test, y_test=None, loss=None):
        '''use to predict data or estimate the accurary.
        Arguments:
            x_test {ndarray}
        Keyword Arguments:
            y_test {ndarray} -- if it's None,the function will predict;or it will test (default: {None})
        '''
        y_predict, _ = self.forwardCalculation(x_test)
        if y_test is None:
            return y_predict
        if not y_test is None:
            assert len(x_test) == len(y_test)
            if loss is None:
                loss = MSE()
            err = loss.get_loss(y_predict, y_test, return_accu=False)
            return err


def data_generator(x, y, batch_size=1):
    '''data generator.Used to create batch train data in format.
        When use GD,set batch_size==len(x)
        When use SGD, set batch_size==1 (default setting)

    Arguments:
        x {ndarray} -- input
        y {ndarray} -- output

    Keyword Arguments:
        batch_size {int} -- batch size (default: {1})
    '''
    assert isinstance(batch_size, int)
    assert len(x) == len(y)
    batch_num = (len(x) + batch_size - 1) // batch_size
    while (True):
        for i in range(batch_num):
            X = x[i * batch_size:(i + 1) * batch_size]
            Y = y[i * batch_size:(i + 1) * batch_size]
            yield (X, Y)