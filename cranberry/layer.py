import numpy as np
from cranberry.tensor import Tensor

class Function(object):

  def forward(self):
    raise NotImplementedError

  def backward(self):
    raise NotImplementedError

  def get_parameters(self):
    return []

class Linear(Function):

  def __init__(self, in_size, out_size):
    self.w = Tensor((in_size, out_size))
    self.b = Tensor((1, out_size))
    self.type = 'linear'

  def forward(self, x):
    output = np.dot(x, self.w.data) + self.b.data
    self.input = x
    return output

  def backward(self, d_y):
    self.w.grad += np.dot(self.input.T, d_y)
    self.b.grad += np.sum(d_y, axis = 0, keepdims = True)
    grad_input = np.dot(d_y, self.w.data.T)
    return grad_input

  def get_parameters(self):
    return [self.w, self.b]

class Normalize(Function):

  """
  TODO: This is incomplete: need to introduce proper batch normalization as per https://arxiv.org/pdf/1502.03167v1.pdf
  """

  def __init__(self, epsilon = 1e-8):
    self.type = 'normalization'
    self.epsilon = epsilon

  def forward(self, x):
    self.gamma = Tensor((x.shape))
    self.beta = Tensor((1, x.shape[0]))
    self.m = x.shape[0]
    self.mean = np.sum(x, axis = 0, keepdims = True) / self.m
    self.var = np.sum(np.square(x - self.mean), axis = 0, keepdims = True) / self.m
    x_norm = np.divide(x - self.mean, np.sqrt(self.var + self.epsilon))


  def backward(self, d_y):
    d_x = np.add(np.multiply(d_y, self.std), self.mean)
    return d_x

  def get_parameters(self):
    return []

class ReLU(Function):
    
  def __init__(self, inplace = True):
    self.type = 'activation'
    self.inplace = inplace 

  def forward(self, x):
    if self.inplace:
      x[x < 0] = 0.
      self.activated = x
    else:
      self.activated = x*(x>0)
    return self.activated

  def backward(self, d_y):
    d_x = d_y * (self.activated > 0)
    return d_x

  def get_parameters(self):
    return []


        