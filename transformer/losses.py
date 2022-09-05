try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

from transformer.activations import Sigmoid, Softmax, ReLU, LogSoftmax


class MSE():

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return np.power(t - y, 2)

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return -2 * (t - y) / np.prod(y.shape[1:])


class BinaryCrossEntropy():

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return -(t * np.log(y + 1e-8) + (1 - t) * np.log(1 - y + 1e-8))

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return -t / (y + 1e-8) + (1 - t) / (1 - (y + 1e-8))


class CategoricalCrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return np.where(t == self.ignore_index, 0, - t * np.log(y))

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return np.where(t == self.ignore_index, 0, -t / y)


class CrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index
        self.log_softmax = LogSoftmax()

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        log_softmax = self.log_softmax.forward(y)
        nll_loss = -log_softmax[np.arange(len(t)), t]
        
        return np.where(t == self.ignore_index, 0, nll_loss)

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        batch_size = y.shape[0]
        err = 1/batch_size
        nll_loss_der = -1 * np.where(np.isin(y, y[np.arange(len(t)), t]), err, 0).astype(y.dtype)
       
        output_err = self.log_softmax.backward(nll_loss_der)
        
        return np.where(t.reshape(-1, 1) == self.ignore_index, 0, output_err)



# import torch
# import torch.nn as nn
# class TorchCrossEntropy():
#     def __init__(self, ignore_index = None) -> None:
#         self.ignore_index = ignore_index
#         if ignore_index is not None:
#             self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
#         else:
#             self.criterion = nn.CrossEntropyLoss()
        

#     def loss(self, y, t):
#         y = torch.tensor(y, requires_grad=True)
#         t = torch.tensor(t.flatten(), requires_grad=False)
#         # print(y.shape, t.shape)
#         self.torch_loss = self.criterion(
#             y,  # (batch_size * (target_seq_len - 1), vocab_size)
#             t # (batch_size * (target_seq_len - 1))
#         )
#         # print(type(self.torch_loss))
#         return self.torch_loss.data.numpy()

#     def derivative(self, y, t):
#         y = torch.tensor(y, requires_grad=True)
#         t_shape = t.shape
#         t = torch.tensor(t.flatten(), requires_grad=False)
#         t.reshape(t.shape)
#         # print(y.shape, t.shape)
#         self.torch_loss = self.criterion(
#             y,  # (batch_size * (target_seq_len - 1), vocab_size)
#             t # (batch_size * (target_seq_len - 1))
#         )
#         grad = torch.autograd.grad(self.torch_loss, y, retain_graph=True)
        
#         return grad[0].data.numpy()




loss_functions = {
    
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy()

}