import numpy as np
from random import shuffle
import NeNe
import NeNe.Activation as activation
from NeNe.Loss import *
from NeNe.Layer import *
from NeNe.LRD import *


def load_data(path):
    '''load breast-cancer-wisconsin.data from files.with preprocessing(convert label into one-hot output format.)
    Arguments:
        path {str} -- the path of the file
    Returns:
        num_input, num_output, num_sample, x, y_onehot[tuple]
    '''
    # read files
    with open(path, mode='r') as f:
        raw = f.readlines()
    # get shape param
    length = len(raw)
    n_features = len(raw[0].split(',')) - 2
    # convert into x,y with deleting fatel value
    i = 0
    x, y = list(), list()
    for single in raw:
        single_split = single.split(',')
        try:
            x.append(np.array(single_split[1:-1], dtype=np.int))
            y.append(np.int(single_split[-1]))
            i += 1
        except:
            continue
    x, y = np.array(x), np.array(y)
    # one-hot y sequences
    y_max = int(np.max(y) + 1)
    y_onehot = np.zeros((len(y), y_max))
    for (i, single_y) in enumerate(y):
        y_onehot[i, single_y] = 1
    return n_features, y_max, length, x, y_onehot


def normalization(x):
    this_x = x.copy().astype(np.float)
    n_features = this_x.shape[-1]
    for i in range(n_features):
        this_mean = np.mean(this_x[:, i])
        this_sd = np.std(this_x[:, i])
        this_x[:, i] = (this_x[:, i] - this_mean) / this_sd
    return this_x


def data_split(x, y, split_rate=0.1):
    split_index = int(len(x) * split_rate)
    x1 = x[:split_index]
    x2 = x[split_index:]
    y1 = y[:split_index]
    y2 = y[split_index:]
    return x1, y1, x2, y2


def data_shuffle(x, y):
    index = [i for i in range(len(x))]
    shuffle(index)
    x = x[index]
    y = y[index]
    return x, y


def main():
    # data preprocessing
    num_input, num_output, num_sample, x, y = load_data(
        'BO\\breast-cancer-wisconsin.data')
    x = normalization(x)
    x, y = data_shuffle(x, y)

    print('input data structure:', num_input, '*', num_sample)
    print('label structure:', num_output, '*', num_sample)

    x_train, y_train, x_test, y_test = data_split(x, y, split_rate=0.8)
    x_train, y_train, x_valid, y_valid = data_split(x_train, y_train, split_rate=0.8)

    # model building
    model = NeNe.NeNe()
    model.add(InputLayer(num=num_input))
    model.add(Layer(num=500, activation=activation.Tanh(), init_seed='norm'))
    model.add(Layer(num=500, activation=activation.Tanh(), init_seed='norm'))
    model.add(Layer(num=500, activation=activation.Tanh(), init_seed='norm'))
    model.add(Layer(num=500, activation=activation.Tanh(), init_seed='norm'))
    model.add(
        Layer(
            num=num_output, activation=activation.SoftMax(), init_seed='norm'))
    model.summary()

    model.fit(
        train_data=(x_train, y_train),
        valid_data=(x_valid, y_valid),
        epoch=20,
        lrf=LRD_cooldown(lr=1.0, epoch_wait=3, decay_rate=0.1),
        batch_size=5,
        loss=MSE(),
        accu_echo=True)
    loss, accu = model.predict(x_test, y_test, loss=MSE(), get_accu=True)
    print('\nmodel on test:loss=%.4f;accu=%.2f%%' % (loss, accu * 100))


main()