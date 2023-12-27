import numpy as np
from Layer import BaseActivation, BaseLayer

class Network:
    
    def __init__(self, data):
        self.layers = list()
        self.data = data
        # self.weights = list()
        # self.bias = list()

    def add(self, layer, activation:BaseActivation = None):
        self.layer = layer
        self.layers.append(layer)
        # print(f'Layer: {type(layer)}')
        # self.add_weights()
        # print(f'Weights: {self.layer.weights}')
        # self.add_bias()
        # print(f'Bias: {self.layer.bias}')
        if activation != None:
            self.layers.append(activation)

    # def add_weights(self):
    #     # if len(self.layers) <= 1:
    #     #     return
    #     new_weights = np.random.normal(loc=0, scale=1, size=(self.layers[-1].n_nodes, ))
    #     self.layer.weights.append(new_weights)
        
    # def add_bias(self):
    #     # if len(self.layers) <= 1:
    #     #     return
    #     new_bias = np.random.normal(loc=0, scale=1, size=(1))
    #     self.layer.bias.append(new_bias)
            
    def compile(self):
        i=0
        while i<=len(self.layers)-1:
            if isinstance(self.layers[i], BaseLayer):
                if i==0: #FIRST DENSE LAYER
                    weights = np.random.normal(loc=0, scale=1, size=(self.layers[i].n_nodes,self.layers[i+2].n_nodes))
                    self.layers[i].weights.append(weights)
                    bias = np.random.normal(loc=0, scale=1, size=(1))
                    self.layers[i].bias.append(bias)
                elif i+2 <= len(self.layers)-1: #MIDDLE LAYERS
                    weights = np.random.normal(loc=0, scale=1, size=(self.layers[i].n_nodes,self.layers[i+2].n_nodes))
                    self.layers[i].weights.append(weights)
                    bias = np.random.normal(loc=0, scale=1, size=(1))
                    self.layers[i].bias.append(bias)
                elif self.layers[i] == self.layers[-2]: #LAST DENSE LAYER
                    pass
               
                print(f'Dense #: {i%2}\n Layer Type: {self.layers[i]}\n, Weights: {self.layers[i].weights}\n,Bias: {self.layers[i].bias}, Activation: {self.layers[i+1]}')
            i = i+1

    def forward(self):
        intermediate = self.data
        for layer in self.layers:
            intermediate = layer.forward(intermediate)
        return intermediate

