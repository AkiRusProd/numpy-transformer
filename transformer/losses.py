import numpy as np
from transformer.activations import Softmax


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


class CrossEntropy():
    def __init__(self, ignore_index) -> None:
        self.ignore_index = ignore_index
        self.softmax = Softmax()

    def loss(self, y, t):
        m = t.shape[0]
        p = self.softmax.function(y)
        log_likelihood = -np.log(p[range(m),t])
        loss = np.sum(log_likelihood) / m
        return np.where(t == self.ignore_index, 0, loss)

    def derivative(self, y, t):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = t.shape[0]
        grad = self.softmax.function(y)
        grad[range(m),t] -= 1
        grad = grad/m
        return np.where(t == self.ignore_index, 0, grad)


import torch
import torch.nn as nn
class TorchCrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index
        if ignore_index is not None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss()
        

    def loss(self, y, t):
        y = torch.tensor(y, requires_grad=True)
        t = torch.tensor(t.flatten(), requires_grad=False)
        # print(y.shape, t.shape)
        self.torch_loss = self.criterion(
            y,  # (batch_size * (target_seq_len - 1), vocab_size)
            t # (batch_size * (target_seq_len - 1))
        )
        # print(type(self.torch_loss))
        return self.torch_loss.data.numpy()

    def derivative(self, y, t):
        y = torch.tensor(y, requires_grad=True)
        t_shape = t.shape
        t = torch.tensor(t.flatten(), requires_grad=False)
        t.reshape(t.shape)
        # print(y.shape, t.shape)
        self.torch_loss = self.criterion(
            y,  # (batch_size * (target_seq_len - 1), vocab_size)
            t # (batch_size * (target_seq_len - 1))
        )
        grad = torch.autograd.grad(self.torch_loss, y, retain_graph=True)
        
        return grad[0].data.numpy()




loss_functions = {
    
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy()

}