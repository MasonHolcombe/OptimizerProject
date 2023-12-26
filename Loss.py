import numpy as np

class BaseLoss():
    def loss_func():
        pass
    def loss_grad():
        pass

class Categorical_CE(BaseLoss):
    def loss_func(y_true, y_pred):
        return -1 * np.dot(np.log(y_pred + 10**-100, y_true))
    def loss_grad(y_true, y_pred):
        return -1 * (y_true/(y_pred + 10**-100))
    


    