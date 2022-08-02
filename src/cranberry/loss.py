import numpy as np
from cranberry.layer import Function

class SoftmaxWithLoss(Function):

  def __init__(self):
    self.type = 'normalization'

  def forward(self, x, t):
    unnormalized_prob = np.exp(x-np.max(x, axis=1, keepdims=True))
    self.prob = unnormalized_prob / np.sum(unnormalized_prob, axis=1, keepdims=True)
    self.t = t
    self.loss = -np.log(self.prob[np.arange(self.prob.shape[0]), self.t])
    return self.loss

  def backward(self):
    grad = self.prob.copy()
    grad[np.arange(self.prob.shape[0]), self.t] -= 1.0
    grad /= self.prob.shape[0]
    return grad

class Softmax(Function):

  def __init__(self):
    self.type = 'normalization'

  def forward(self, x):
    unnormalized_prob = np.exp(x-np.max(x, axis=1, keepdims=True))
    self.prob = unnormalized_prob / np.sum(unnormalized_prob, axis=1, keepdims=True)
    return self.prob

  def backward(self):
    return self.prob

class CrossEntropy(Function):

  def __init__(self):
    self.type = 'normalization'

  def forward(self, x, t):
    self.t = t
    self.loss = -np.log(x[np.arange(x.shape[0]), self.t])
    return self.loss

  def backward(self):
    grad = self.prob.copy()
    grad[np.arange(self.prob.shape[0]), self.t] -= 1.0
    grad /= self.prob.shape[0]
    return grad