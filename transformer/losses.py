import numpy as np


class MSE():

    def loss(self, y, t):

        return np.power(t - y, 2)

    def derivative(self, y, t):

        return -(t - y)


class BinaryCrossEntropy():

    def loss(self, y, t):

        return -(t * np.log(y + 1e-8) + (1 - t) * np.log(1 - y + 1e-8))

    def derivative(self, y, t):

        return -t / (y + 1e-8) + (1 - t) / (1 - (y + 1e-8))


class CategoricalCrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index

    def loss(self, y, t):
        
        return np.where(t == self.ignore_index, 0, - t * np.log(y))

    def derivative(self, y, t):

        return np.where(t == self.ignore_index, 0, -t / y)


loss_functions = {
    
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy()

}