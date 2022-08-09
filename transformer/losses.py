import numpy as np
from transformer.activations import Sigmoid, Softmax, ReLU, LogSoftmax


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
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index
        # self.softmax = Softmax()
        self.log_softmax = LogSoftmax()
        # self.af = ReLU()

    def loss(self, y, t):
        log_softmax = self.log_softmax.function(y) #np.log(self.softmax.function(y))
        nll_loss = -log_softmax[np.arange(len(t)), t]
        
        return np.where(t == self.ignore_index, 0, nll_loss)

    def derivative(self, y, t):
        batch_size = y.shape[0]
        err = 1/batch_size
        nll_loss_der = -1 * np.where(np.isin(y, y[np.arange(len(t)), t]), err, 0)
       
        output_err = self.log_softmax.jacobian_derivative(y, nll_loss_der)
        
        return np.where(t.reshape(-1, 1) == self.ignore_index, 0, output_err)

    # def loss(self, y, t):
    #     softmax = self.softmax.function(y)
    #     nll_loss = -np.log(softmax[np.arange(len(t)), t])

    #     return np.where(t == self.ignore_index, 0, nll_loss)

    # def derivative(self, y, t):
    #     batch_size = y.shape[0]
    #     err = 1/batch_size
    #     nll_loss_der = -1 * np.where(np.isin(y, y[np.arange(len(t)), t]), err, 0) #OK

    #     #dlog(S(x))/d x = 1/S(x) * dS(x)/dx = 1/S(x) * S(x) * (1 - S(x)) = 1 - S(x)
    #     # log_softmax_der = 1 / self.softmax.function(nll_loss_der) * self.softmax.derivative(nll_loss_der) # = 1 - self.softmax.function(y)
    #     log_softmax_der = 1 - self.softmax.function(nll_loss_der)
        
    #     #compute nll loss derivative
    #     # return np.where(t == self.ignore_index, 0, grad)
    #     # return np.where(t == self.ignore_index, 0, -self.softmax.derivative(y)[np.arange(len(t)), t])
    #     # return np.where(t == self.ignore_index, 0, log_softmax_der)
    #     return log_softmax_der



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