import numpy as np

class BaseLayer:

    def __init__(self, n_nodes:int):
        self.n_nodes = n_nodes

    def forward(self, data):
        pass
    
    def backward(self, idx, dx):
        pass

class Dense(BaseLayer):

    def __init__(self, n_nodes:int, lr_rate):
        super().__init__(n_nodes)
        self.weights = None
        self.bias = None
        self.lr_rate = lr_rate
    
    def forward(self, data):
        self.data = data
        if self.weights is None or self.bias is None:
            raise Exception('Weights and/or Bias not initialized')
        return np.dot(data, self.weights) + self.bias

    def backward(self, idx, dx):
        cur_sample = self.data[idx]
        print(f"shape of derivative: {dx.shape}")
        print(f"shape of sample: {cur_sample.shape}")
        print(f"shape of weight: {self.weights.shape}")
        print(f"shape of bias: {self.bias.shape}")
        dw = np.dot(self.weights, dx.T)
        db = np.sum(dx, axis=0)
        self.weights = self.weights - np.dot(self.lr_rate, dw)
        self.bias = self.bias - np.dot(self.lr_rate, db)
        res = np.dot(dx, self.weights.T)
        return res
    
class Input(Dense):
    def forward(self, data):
        self.data = data
        return self.data.T
    def backward(self,idx,  dx):
        return dx
    



class BaseActivation():
    # do something
    def forward(self, vec):
        pass
    def backward(self, idx, vec):
        pass

class ReLU(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        self.out = np.maximum(np.zeros(shape=vec.shape), vec)
        return self.out
    
    def backward(self, idx, dx):
        cur_sample = self.out[idx]
        print(f"self.out shape: {cur_sample.shape}")
        print(f"dx shape: {dx.shape}")
        cur_dx = np.greater(cur_sample.T, np.zeros(shape=cur_sample.T.shape)).astype(int)
        return dx * cur_dx
    
class Sigmoid(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        self.out = 1 / (1 + np.exp(-vec))
        return self.out
    def backward(self, idx, dx):
        res = dx * self.out * (1 - self.out)
        return res
    
class Linear(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        self.out = vec
        return self.out
    def backward(self, idx, dx):
        return dx * np.ones(shape=self.out.shape)
    
class Softmax(BaseActivation):
    def __init__(self):
        pass
    def forward(self, vec):
        exps = np.exp(vec)
        self.out = exps / np.reshape(np.sum(exps, axis=1), (-1,1))
        # print(f"Softmax shape: {self.out.shape}")

        return self.out
    def backward(self, idx, dx):
        # print(f'dx: {dx}')
        n = self.out.shape[1]
        jacobian = np.zeros(shape=(n,n))
        cur_sample = self.out[idx]
        for i in range(n):
            for j in range(n):
                # print(i,j)
                # print(j)
                kronecker = 1 if i == j else 0
                jacobian[i][j] = cur_sample[i] * (kronecker - cur_sample[j]) 
        return np.dot(dx, jacobian)
    
    # def individual_back(self,individual):
    #     n = len(individual)
    #     jacobian = np.zeros(shape=(n,n))
    #     for i in range(n):
    #         for j in range(n):
    #             kronecker = 1 if i == j else 0
    #             print(individual.shape)
    #             jacobian[i][j] = individual[i]* (kronecker - individual[j]) 
    #     return individual