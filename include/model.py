import numpy as np

class Model():

    def __init__(self):
        self.functions = []
        self.parameters = []
        self.loss = None
        self.loss_history = []

    def add(self, function):
        self.functions.append(function)
        self.parameters += function.get_parameters()

    def __initialize_network(self, scale = 0.01):
        for f in self.functions:
            if f.type == 'linear':
                f.w.initialize_random(scale)
                f.b.initialize_zero()

    def forward(self, X):
        for f in self.functions:
            X = f.forward(X)
        return X

    def backward(self, d_loss):
        for f in self.functions[::-1]:
            d_loss = f.backward(d_loss)
        return d_loss

    def fit(self, X ,Y, batch_size = 1, epochs = 1, optimizer = None, loss_func = None, lr = 0.01, weight_decay = 0.0, momentum = 0.0, initialize = True):
        if initialize:
            self.__initialize_network()
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                X_forward = self.forward(X_batch)
                loss, d_loss = loss_func.forward(X_forward, Y_batch), loss_func.backward()
                self.backward(d_loss)
                optimizer.step()
                self.loss_history.append(self.loss)

        return loss_history
    
    def predict(self, X):
        for f in self.functions:
            X = f.forward(X)
        return X

    def save(self, filename):
        np.savez_compressed(filename, function = self.functions, parameters = self.parameters, loss = self.loss)

    def load(self, filename):
        npz = np.load(filename)
        self.functions = npz['function']
        self.parameters = npz['parameters']
        self.loss = npz['loss']
        return self