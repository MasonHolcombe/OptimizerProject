import numpy as np

class BaseLayer:

    def __init__(self, n_nodes:int):
        self.n_nodes = n_nodes

    def forward(self, data):
        pass
    
    def backward(self, dx):
        pass

class Dense(BaseLayer):

    def __init__(self, n_nodes:int, lr_rate):
        super().__init__(n_nodes)
        self.weights = list()
        self.bias = list()
        self.lr_rate = lr_rate
    
    def forward(self, data):
        self.data = data
        # if self.weights == [] or self.bias == []:
        #     raise Exception('Weights and/or Bias not initialized')
        return np.dot(data, self.weights) + self.bias

    def backward(self, dx):
        dw = np.dot(self.data.T, dx)
        db = np.sum(dx, axis=0)
        self.weights = self.weights - np.dot(self.lr_rate, dw)
        self.bias = self.bias - np.dot(self.lr_rate, db)
        res = np.dot(dx, self.weights.T)
        return res


class BaseActivation():
    # do something
    def forward(self, vec):
        pass
    def backward(self, vec):
        pass

class ReLU(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        self.out = np.maximum(np.zeros(shape=vec.shape), vec)
        return self.out
    
    def backward(self, dx):
        return dx * np.greater(self.out, np.zeros(shape=self.out.shape)).astype(int)
    
class Sigmoid(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        self.out = 1 / (1 + np.exp(-vec))
        return self.out
    def backward(self, dx):
        res = dx * self.out * (1 - self.out)
        return res
    
class Linear(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        self.out = vec
        return self.out
    def backward(self, dx):
        return dx * np.ones(shape=self.out.shape)
    
class Softmax(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        exps = np.exp(vec)
        return exps / np.sum(exps)
    def backward(self, vec):
        pass



