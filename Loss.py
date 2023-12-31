import numpy as np

class BaseLoss():
    def forward():
        pass
    def backward():
        pass

class Categorical_CE(BaseLoss):
    def forward(self, y_true, y_pred):
        self.probs = y_pred[np.arange(y_pred.shape[0]), y_true]
        return np.sum(-np.log(self.probs * 10**-100))
    def backward(self):
        return -1/self.probs#(y_true/(y_pred + 10**-100))
    


    