from Layer import BaseActivation, BaseLayer, Dense
from Loss import BaseLoss
import numpy as np

class Network:
    
    def __init__(self, data, loss):
        self.layers = list()
        self.data = data
        self.loss = loss
    
    def __repr__(self):
        ret = str()
        ret += 'Displaying Network\n'
        for i, layer in enumerate(self.layers):
            ret += f"Layer {i}.\n\ttype: {type(layer)}\n"
            if isinstance(layer,Dense):
                if i != 0:
                    ret += f"\tShape {layer.weights.shape}\n"
        return ret

    def add(self, layer, activation:BaseActivation = None):
        self.layers.append(layer)
        if activation != None:
            self.layers.append(activation)


    def compile(self, debug=0):
        for i, layer in reversed(list(enumerate(self.layers))[1:]): #LOOP THROUGH ALL LAYERS AND EXCLUDE INPUT LAYER
            if isinstance(layer, Dense):
                prevDense = self.layers[self.findPreviousDense(i)]
                #self.layers[i]
                layer.weights = np.random.normal(loc=0, scale=1, size=(prevDense.n_nodes,layer.n_nodes))
                #self.layers[i]
                layer.bias = np.random.normal(loc=0, scale=1, size=(1, layer.n_nodes))
                if debug == 1:
                    print(f'Dense Layer #: {int(i)}\nLayer Type: {type(layer)}\nWeights: {layer.weights}\nBias: {layer.bias}\n')
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                
    def findPreviousDense(self, i):
        for j in range(i-1, -1, -1):
            if isinstance(self.layers[j], Dense):
                return j
        raise Exception(f'No Dense Layer Found Before Layer {i}')

    def forward(self):
        intermediate = self.data
        for layer in self.layers:
            intermediate = layer.forward(intermediate)
        return intermediate
    
    def calcLoss(self, y_true, y_pred):
        loss = self.loss.forward(y_true, y_pred)
        return loss


    def backward(self):
        all_dx = self.loss.backward()
        for i, sample in enumerate(all_dx):
            dx = sample
            # print(f"ran {i}")
            for layer in reversed(self.layers):
                dx = layer.backward(i, dx)
                print(layer)
            