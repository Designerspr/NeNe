'''Provides some learning rate decay function.
'''
class LRD(object):
    def __init__():
        return
    def update(lr):
        return lr

class LRD_exp_decay(object):
    '''the exp decay of the learning rate.
    To define a LRD_exp_decay,3 args needed:
    lr {float} :initial learning rate
    decay rate {float,0<rate<1}: the rate
    decay_epoch {int}: the period.
    learning rate would update like:

    lr(this epoch)=lr*decay_rate**(epoch/decay_epoch)
    '''
    def __init__(lr,decay_rate,decay_epoch):
        self.init_lr=lr
        self.decay_rate=decay_rate
        self.decay_epoch=decay_epoch

