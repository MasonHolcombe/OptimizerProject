import numpy as np
from Layer import Layer

class Network:
    
    def __init__(self, data):
        self.layers = list()
        self.data = data

    def add(self, layer):
        self.layers.append(layer)

    def forward(self):
        output = list()
        intermediate = self.data
        for i, layer in enumerate(self.layers):
            intermediate = intermediate * layer.weights
            activation_func = np.vectorize(layer.activation.activate)
            intermediate = activation_func(intermediate)
            
