import numpy as np

class BaseActivation:
    # do something
    def activate(self, vec):
        pass
    def derivative(self, vec):
        pass

class ReLU(BaseActivation):
    def __init__(self):
        pass
    def activate(self, vec):
        return np.maximum(np.zeros(shape=vec.shape), vec)
    
    def derivative(self, vec):
        return np.greater(vec, np.zeros(shape=vec.shape)).astype(int)
        
class Sigmoid(BaseActivation):
    def __init__(self):
        pass
    def activate(self, vec):
        return 1 / (1 + np.exp(-vec))
    def derivative(self, vec):
        return self.activate(vec) * (1 - self.activate(vec))
    
class Linear(BaseActivation):
    def __init__(self):
        pass
    def activate(self, vec):
        return vec
    def derivative(self, vec):
        return np.ones(shape=vec.shape)
    
class Softmax():
    def __init__(self):
        pass
    def activate(self, vec):
        exps = np.exp(vec)
        return exps / np.sum(exps)
    def derivative(self, vec):
        # for val in vec:
