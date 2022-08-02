import numpy as np

class Optimizer(object):

  def __init__(self, parameters):
    self.parameters = parameters

  def step(self):
    raise NotImplementedError

  def zero_grad(self):
    for p in self.parameters:
      p.grad = 0.

class SGD(Optimizer):
  
  def __init__(self, parameters, lr = 0.01, weight_decay = 0.0, momentum = 0.0):
    super().__init__(parameters)
    self.lr = lr
    self.weight_decay = weight_decay
    self.momentum = momentum
    self.velocity = []
    for p in self.parameters:
      self.velocity.append(np.zeros_like(p.grad))
  
  def step(self):
    for p, v in zip(self.parameters, self.velocity):
      v = self.momentum * v - self.lr * (p.grad + self.weight_decay * p.data)
      p.data += v

class Adam(Optimizer):

  def __init__(self, parameters, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
    super().__init__(parameters)
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.m = []
    self.v = []
    for p in self.parameters:
      self.m.append(np.zeros_like(p.grad))
      self.v.append(np.zeros_like(p.grad))
    
  def step(self):
    for p, m, v in zip(self.parameters, self.m, self.v):
      m = self.beta1 * m + (1 - self.beta1) * p.grad
      v = self.beta2 * v + (1 - self.beta2) * p.grad ** 2
      m_hat = m / (1 - self.beta1 ** self.t)
      v_hat = v / (1 - self.beta2 ** self.t)
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
      self.t += 1