from Activations import BaseActivation
import numpy as np

class Layer:

    def __init__(self, n_nodes:int, activation:BaseActivation):
        self.n_nodes = n_nodes
        self.activation = activation

