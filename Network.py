import numpy as np
from Layer import Layer

class Network:
    
    def __init__(self, data):
        self.layers = list()
        self.data = data
        self.weights = list()
        self.bias = list()

    def add(self, layer):
        self.layers.append(layer)
        self.add_weights()
        self.add_bias()

    def add_weights(self):
        if len(self.layers) <= 1:
            return
        new_weights = np.random.normal(loc=0, scale=1, size=(self.layers[-2].n_nodes, self.layers[-1].n_nodes))
        self.weights.append(new_weights)
        
    def add_bias(self):
        if len(self.layers) <= 1:
            return
        new_bias = np.random.normal(loc=0, scale=1, size=(self.layers[-1].n_nodes, ))
        self.bias.append(new_bias)

    def forward(self):
        intermediate = self.data
        for i, weight in enumerate(self.weights):
            intermediate = np.dot(intermediate, weight) + self.bias[i]
            intermediate = self.layers[i+1].activation.activate(intermediate)
        return intermediate
        
