'''Provides some learning rate decay function.
'''
import numpy as np


class LRC(object):
    '''learning rate function with constant learning rate.
    when do not want to choose any LRD functions, use this one.
    Learning rate does not change during the training.
    Requires one args only when initializing: learning rate.
    '''

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, epoch, err):
        return self.lr


class LRD_expdecay(LRC):
    '''the exp decay of the learning rate.
    To define a LRD_exp_decay,3 args needed:
    lr {float} :initial learning rate
    decay rate {float,0<rate<1}: the rate
    decay_epoch {int}: the period.
    learning rate would update like:

    lr(this epoch)=lr*decay_rate**(epoch/decay_epoch)
    '''

    def __init__(self, lr=0.01, decay_rate=0.9, decay_epoch=1,
                 staircase=False):
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
        self.staircase = staircase

    def update(self, epoch, err):
        if self.staircase:
            value = self.lr * self.decay_rate**int(epoch / self.decay_epoch)
            if int(epoch / self.decay_epoch) != int(
                (epoch - 1) / self.decay_epoch):
                print('\nlearning rate update to', value)
            return value
        else:
            return self.lr * self.decay_rate**(epoch / self.decay_epoch)


class LRD_cooldown(LRC):
    '''this LRD function will decrease the lr if err do not decrease.
    '''

    def __init__(self, lr, epoch_wait=1, decay_rate=0.9):
        self.lr = lr
        self.epoch_max = epoch_wait + 1
        self.decay_rate = decay_rate
        self.err_seq = np.zeros(self.epoch_max)
        self.err_seq[0] = np.inf
        self.epoch_not_down = 1

    def update(self, epoch, err):
        self.err_seq[self.epoch_not_down] = err
        # if error decline
        if np.where(
                np.min(self.err_seq[:self.epoch_not_down + 1]) ==
                self.err_seq)[0][0] == self.epoch_not_down:
            self.err_seq = np.zeros(self.epoch_max)
            self.err_seq[0] = err
            self.epoch_not_down = 1
        else:
            # if not
            self.epoch_not_down += 1
            # if wait too much epoch with no decline
            if self.epoch_not_down == self.epoch_max:
                # decrease learning rate and fresh err_seq
                self.lr *= self.decay_rate
                self.err_seq = np.zeros(self.epoch_max)
                self.err_seq[0] = err
                self.epoch_not_down = 1
                print('learning rate update to:', self.lr)
        return self.lr
