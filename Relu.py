from BaseActivation import BaseActivation

class ReLU(BaseActivation):
    def activate(point):
        return max(0, point)
    
    def derivative(point):
        if point < 0:
            return 0
        else:
            return 1