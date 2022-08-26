import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


# import numpy as np
from transformer.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, CrossEntropy
from transformer.activations import Sigmoid, Softmax, ReLU, LogSoftmax
# # class Softmax():

# #     def forward(self, x):
# #         tmp = np.exp(x - np.max(x, axis=1, keepdims=True))
# #         self.output =  tmp / np.sum(tmp, axis=1, keepdims=True)

# #         return self.output

# #     def backward(self, grad):
# #         n = np.size(self.output)

# #         tmp = np.tile(self.output, n)
# #         print(tmp * (np.identity(n) - np.transpose(tmp)).shape)
# #         return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), grad)


# # softmax = Softmax()

# # print(softmax.forward(np.array([[1, 2, 3]])))
# # print(softmax.backward(np.array([[1, 2, 3]]))
# # )


# # class Softmax():

# #     def function(self, x):
# #         e_x = np.exp(x - np.max(x, axis = -1, keepdims=True))
        
# #         self.softmax =  e_x / np.sum(e_x, axis = -1, keepdims=True)
# #         return self.softmax

# #     def derivative(self, x):# x <- входные данные итак; i=j
# #         f_x = self.function(x)
        
# #         return f_x * (1.0 - f_x)

# #     def derivative2(self, x, grad = None): #неправильно отсносительно моей реализации/ х и есть инпут
# #         #https://e2eml.school/softmax.html
# #         softmax = self.function(x).reshape(1, -1)
# #         # grad = grad.reshape(1, -1)
# #         d_softmax = (softmax * np.identity(softmax.size)   #Jacobian matrix                      
# #                     - softmax.transpose() @ softmax)
# #         #https://suzyahyah.github.io/calculus/machine%20learning/2018/04/04/Jacobian-and-Backpropagation.html
# #         # input_grad = grad @ d_softmax

# #         # return input_grad.reshape(x.shape)
# #         return d_softmax

# #     def derivative3(self, x, grad = None):
# #         softmax = self.function(x)
# #         s = softmax.reshape(-1,1)
# #         # return  (grad.reshape(1, -1) @ (np.diagflat(s) - np.dot(s, s.T))).reshape(x.shape)
# #         return  np.diagflat(s) - np.dot(s, s.T)

# #     # def derivative(self, grad): неправильно отсносительно моей реализации. град и есть инпут; i!=j
# #     #     #https://sgugger.github.io/a-simple-neural-net-in-numpy.html
# #     #     return self.softmax * (grad -(grad * self.softmax).sum(axis=1)[:,None])
# axis = -1

# #2, 2, 2, 1
# activation = Softmax(axis)
# input =  np.array([[1., 2., 3., 4., 5., 6., 7., 8.]]).reshape(2, 2, 2, 1)
# target = np.array([[1., 1., 0., 1., 0., 1., 0., 1.]]).reshape(2, 2, 2, 1)
# # grad = np.array([[2, 3, 4]])

# # der1 = softmax.derivative(np.array([[1, 2, 3]]))
# # print("der1\n", der1)
# # der2 = softmax.derivative2(np.array([[1, 2, 3]]))
# # print("der2\n", der2)
# # der3 = softmax.derivative3(np.array([[1, 2, 3]]))
# # print("der3\n", der3)

# activation_output = activation.function(input)
# print("activation output")
# print(activation_output)
# mse_loss = MSE()
# mse_loss_output = mse_loss.loss(activation_output, target)
# print("mse_loss_output\n", mse_loss_output.mean())
# mse_loss_derivative = mse_loss.derivative(activation_output, target)
# print("mse_loss_derivative\n", mse_loss_derivative)
# # activation_output_derivative = activation.derivative(input) * mse_loss_derivative *2/3
# activation_output_derivative = activation.jacobian_derivative3(input, mse_loss_derivative) #1D OK
# print("activation_output_derivative\n", activation_output_derivative)





# import torch
# import torch.nn as nn

# t_input = torch.tensor(input, requires_grad=True)
# output = nn.Softmax(dim = axis)(t_input) #default dim = -1
# print("torch activation output")
# print(output)
# # log = torch.log(output)
# loss = nn.MSELoss()
# torch_loss = loss(output, torch.tensor(target))
# print(torch_loss)
# torch_loss.backward()
# print(t_input.grad)


# # print(np.sum(input, axis = 1, keepdims=True))
# # print(torch.sum(t_input, dim = 1, keepdim=True))




# # print('TEST')
# # import torch.nn.functional as F

# # a = torch.tensor([1., 2., 3], requires_grad=True)# rand(3, requires_grad=True)
# # print(a.shape)

# # p = F.softmax(a, dim=0) #mb here dim = 1 while its 0 
# # # Specify dim dimensions for sfotmax operation

# # print('softmax:', p)

# # print(torch.autograd.grad(p[2], [a])) #jacobian for element
# # activation = Softmax(axis=1)
# # print(activation.function(a.detach().numpy().reshape(1, -1)))
# # # print(activation.derivative(a.detach().numpy()) * a.detach().numpy())
# # print(activation.derivative2(a.detach().numpy().reshape(1, -1), a.detach().numpy().reshape(1, -1))) #verno
# # Note that loss must be a value of length 1. Here, since p is a value of dim=1 and length is 3, take p[1].

# # print("tst 2d output")
# # print(activation.function(input))
# # print(nn.Softmax(dim = -1)(torch.tensor(input)))



# # # print(np.diagflat(input, axis = -1))
# # softmax = activation.function(input).reshape(1, -1)
# # # grad = grad.reshape(1, -1)
# # d_softmax = (softmax * np.identity(softmax.size)   #Jacobian matrix                      
# #             - softmax.transpose() @ softmax)

# # # print("d_softmax\n", d_softmax)

# # # print(softmax * np.identity(softmax.size))

# # softmax = activation.function(input)
# # d_softmax = softmax[:, :, np.newaxis] * np.tile(np.identity(softmax.shape[1]), (softmax.shape[0], 1, 1)) - softmax.transpose() @ softmax
# # print(d_softmax)
# # # print(np.tile(np.identity(softmax.shape[1]), (softmax.shape[0], 1, 1)))


# # softmax = np.array([[1, 2, 3], [4, 5, 6]])
# # # print(softmax.shape)
# # print(np.einsum('ij, ik -> ijk', softmax, softmax))
# # print(np.matmul(softmax[:, :, None], softmax[:, None, :]))


# # zer_arr = torch.zeros(250, 7600 * 7600)

# # from tempfile import mkdtemp
# # import os.path as path
# # filename = path.join(mkdtemp(), 'newfile.dat')

# # fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
# # print(fp)


# # data = np.arange(12, dtype='float32')
# # data.shape = (3,4)
# # fp[:] = data[:]
# # print(fp)

# # print(fp.filename == path.abspath(filename))

# # fp.flush()

# # newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
# # print(newfp)
# # print(filename)

import numpy as np
import cupy as cp


# x_cpu = np.ones((1000,1000,1000))
# x_gpu = cp.ones((1000,1000,1000))

# logsoftmax = LogSoftmax()

# shape = (2, 3, 10)

# # x = np.arange(0, np.prod(shape)).reshape(shape)
# x = np.random.normal(0, 1, shape)

# x = logsoftmax.forward(x)
# # print(x)
# y = logsoftmax.jacobian_backward(x)  #gradient of x
# print(y.shape)

# y2 = logsoftmax.jacobian_opt_backward(x)  #gradient of x
# print(y2.shape)

# # print(y)
# # print(y2)

# print(np.allclose(y, y2))

# print(np.equal(y, y2))

softmax = Softmax()

shape = (4, 2, 10, 5)

# x = np.arange(0, np.prod(shape)).reshape(shape)
x = np.random.normal(0, 1, shape)

x = softmax.forward(x)
# print(x)
y = softmax.backward(x)  #gradient of x
print(y.shape)

y2 = softmax.backward_opt(x)  #gradient of x
print(y2.shape)

print(y)
print(y2)

print(np.allclose(y, y2))


