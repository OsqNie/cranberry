import numpy as np

class Tensor(object):

    def __init__(self, shape, data=None):
        self.shape = shape
        self.data = data
        if data is None:
            self.data = np.zeros(shape)
        self.grad = np.zeros(shape)

    def initialize_random(self, scale=0.01):
        self.data = np.random.randn(*self.shape) * scale

    def initialize_zero(self):
        self.data = np.zeros(self.shape)