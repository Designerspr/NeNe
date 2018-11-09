import numpy
import Activation

class Layer(object):
    def __init__(self, num_last, num_this, activation=None):
        self.weight = np.zeros((num_last, num_this))
        self.bias=np.zeros(num_this)
        if not activation:
            self.activation=Activation.Linear()
        if isinstance(activation,Activation.Activation):
            raise Exception('invaild activation type: expect Activation.Activation, got',type(activation))
        self.activation=activation
        return

    def calculate(self,value_input):
        value_output=np.multiply(value_input,self.weight)+self.bias
        value_output=self.activation.forward(value_output)
        return value_output
    def train(self,value_output):
        # training bias
        # training weight
        # pass the value to the upper layer