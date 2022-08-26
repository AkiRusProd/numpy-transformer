try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


from numba import njit

"""
References:
    https://mlfromscratch.com/activation-functions-explained/
    https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
"""


class Activation(object):

    def forward(self, x):
        pass

    def backward(self, grad):
        pass


class Sigmoid(Activation):

    def forward(self, x):
        self.x = x

        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        x = self.x
        f_x = self.forward(self.x)

        return grad * (f_x * (1.0 - f_x)).astype(x.dtype)


class Tanh(Activation):

    def forward(self, x):
        self.x = x

        return np.tanh(x)

    def backward(self, grad):
        x = self.x
        return grad * (1.0 - np.power(self.forward(x), 2)).astype(x.dtype)


class Softmax(Activation):
    def __init__(self) -> None:
        self.axis = -1

    def forward(self, x):
        self.x = x
        e_x = np.exp(x - np.max(x, axis = self.axis, keepdims=True))
        
        self.softmax =  e_x / np.sum(e_x, axis = self.axis, keepdims=True)
        return self.softmax

    # def backward(self, x, grad):# i=j
    #     f_x = self.forward(x)
        
    #     return grad * f_x * (1.0 - f_x)
   

    def backward(self, grad = None):
        #https://e2eml.school/softmax.html
        #https://suzyahyah.github.io/calculus/machine%20learning/2018/04/04/Jacobian-and-Backpropagation.html
        #https://sgugger.github.io/a-simple-neural-net-in-numpy.html

        batch_size = self.x.shape[0]
        softmax = self.forward(self.x)
        J = softmax[..., np.newaxis] * np.tile(np.identity(softmax.shape[-1], dtype = self.x.dtype), (softmax.shape[0], *tuple(np.ones(softmax.ndim, dtype = np.int8).tolist()))) - (softmax[..., np.newaxis, :].transpose(*tuple(np.arange(0, softmax.ndim - 1, 1, dtype=np.int8).tolist()), -1, -2) @ softmax[..., np.newaxis, :]) #np.matmul(softmax[:, :, None], softmax[:, None, :])
        input_grad =  grad[..., np.newaxis, :] @ J
        
        return input_grad.reshape(self.x.shape) / batch_size

    # def backward_iter(self, grad = None): #iterative variant
    #     #https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
    #     batch_size = self.x.shape[0]
    #     softmax = self.forward(self.x)

    #     input_grad =  np.zeros(grad.shape, dtype=self.x.dtype)

    #     # for i in range(grad.shape[1]): #1d, 2d
    #     #     sum_val = 0
    #     #     for j in range(grad.shape[1]):
    #     #         sum_val += softmax[:, j] * grad[:, j] * -softmax[:, i]

    #     #     input_grad[:, i] = sum_val

    #     # for i in range(grad.shape[1]):
    #     #     input_grad[:, i] += softmax[:, i] * grad[:, i]

    #     input_grad = self.numba_backward(grad, softmax, input_grad)

    #     return input_grad / batch_size


    # @staticmethod    
    # @njit
    # def numba_backward(grad, softmax, input_grad):

    #     for k in range(grad.shape[0]): #4d
    #         for l in range(grad.shape[1]):
    #             for i in range(grad.shape[3]): 
    #                 sum_val = np.zeros(grad.shape[2], dtype=grad.dtype)
    #                 for j in range(grad.shape[3]):
    #                     sum_val += softmax[k, l, :, j] * grad[k, l, :, j] * -softmax[k, l, :, i]

    #                 input_grad[k, l, :, i] = sum_val

    #             for i in range(grad.shape[3]):
    #                 input_grad[k, l, :, i] += softmax[k, l, :, i] * grad[k, l, :, i]

    #     return input_grad






class LogSoftmax(Activation):
    def __init__(self) -> None:
        self.axis = -1

    def softmax_forward(self, x):
        e_x = np.exp(x - np.max(x, axis = self.axis, keepdims=True))
        
        self.softmax =  e_x / np.sum(e_x, axis = self.axis, keepdims=True)
        return self.softmax

    def forward(self, x):
        self.x = x
        self.log_softmax = np.log(self.softmax_forward(x))
        return self.log_softmax

    # def backward(self, x):# for i==j
    #     f_x = self.forward(x)
        
    #     return (1.0 - f_x)

    # def jacobian_backward(self, grad = None):
    #     batch_size = self.x.shape[0]
    #     softmax = self.softmax_forward(self.x)

    #     J = np.tile(np.identity(softmax.shape[-1], dtype = self.x.dtype), (softmax.shape[0], *np.ones(softmax.ndim, dtype = np.int8))) - (np.ones((*self.x.shape, 1)).astype(np.float32) @ softmax[..., np.newaxis, :])

    #     input_grad =  grad[..., np.newaxis, :] @ J
        
    #     return input_grad.reshape(self.x.shape) / batch_size

    # def jacobian_backward(self, grad = None): #iterative variant
    #     #https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
    #     batch_size = self.x.shape[0]
    #     softmax = self.softmax_forward(self.x)

    #     input_grad =  np.zeros(grad.shape, dtype=self.x.dtype)
        
    #     # for i in range(grad.shape[1]): # for 1d array (1, D)
    #     #     input_grad[:, i] = grad[:, i] - softmax[:, i] * grad[0, :].sum()

    #     # for i in range(grad.shape[0]): # for 2d array (N, D)
    #     #     for j in range(grad.shape[1]):
    #     #         input_grad[i, j] = grad[i, j] - softmax[i, j] * grad[i, :].sum()

    #     # for i in range(grad.shape[0]): #3d array (N, D1, D2)
    #     #     for j in range(grad.shape[1]):
    #     #         for k in range(grad.shape[2]):
    #     #             input_grad[i, j, k] = grad[i, j, k] - softmax[i, j, k] * grad[i, j, :].sum()

    #     input_grad = self.numba_jacobian_backward(grad, softmax, input_grad) #(N, D)
    #     # input_grad = np.asarray(self.numba_jacobian_backward(np.asnumpy(grad), np.asnumpy(softmax), np.asnumpy(input_grad))) #(N, D)
        
    #     return input_grad / batch_size

    # @staticmethod
    # @njit
    # def numba_jacobian_backward(grad, softmax, input_grad):
    #     for i in range(grad.shape[0]): # for 2d array (N, D)
    #         for j in range(grad.shape[1]):
    #             input_grad[i, j] = grad[i, j] - softmax[i, j] * grad[i, :].sum()

    #     return input_grad

    def jacobian_backward(self, grad = None):
        batch_size = self.x.shape[0]
        softmax = self.softmax_forward(self.x)

        input_grad = grad - softmax * grad.sum(axis = self.axis, keepdims=True)

        return input_grad / batch_size




class Softplus(Activation):

    def forward(self, x):
        self.x = x
        return np.log(1 + np.exp(x))

    def backward(self, grad):
        x = self.x
        return grad * 1 / (1 + np.exp(-x))

class Softsign(Activation):

    def forward(self, x):
        self.x = x
        return x / (1 + np.abs(x))

    def backward(self, grad):
        x = self.x
        return grad * 1 / np.power(1 + np.abs(x), 2).astype(x.dtype)

class Swish(Activation):

    def __init__(self, beta = 1):
        self.beta = beta

    def forward(self, x):
        self.x = x
        self.sigmoid = lambda z: 1 / (1 + np.exp(-z)) 

        return x * self.sigmoid(self.beta * x)

    def backward(self, grad):
        x = self.x
        f_x = self.forward(self.x)

        return grad * (self.beta * f_x + self.sigmoid(self.beta * x) * (1 - self.beta * f_x)).astype(x.dtype)

class Mish(Activation):

    def forward(self, x):
        self.x = x
        return x * np.tanh(np.log(1 + np.exp(x)))

    def backward(self, grad):
        x = self.x
        return grad * (np.exp(x) * (4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)) / np.power((2 * np.exp(x) + np.exp(2 * x) + 2), 2)).astype(x.dtype)

class TanhExp(Activation):

    def forward(self, x):
        self.x = x

        return x * np.tanh(np.exp(x))

    def backward(self, grad):
        x = self.x
        return grad * (np.tanh(np.exp(x)) - x * np.exp(x) * (np.power(np.tanh(np.exp(x)), 2) - 1)).astype(x.dtype)


class ReLU(Activation):

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, 0, 1).astype(x.dtype)


class LeakyReLU(Activation):

    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x <= 0, self.alpha * x, x)

    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, self.alpha, 1).astype(x.dtype)


class ELU(Activation):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha 

    def forward(self, x):
        self.x = x
        return np.where(x <= 0, self.alpha * (np.exp(x) - 1), x)

    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, self.alpha + self.forward(x), 1).astype(x.dtype)


class SELU(Activation):

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.lmbda = 1.0507009873554804934193349852946 

    def forward(self, x):
        self.x = x
        return self.lmbda * np.where(x > 0, x, self.alpha*(np.exp(x)-1))

    def backward(self, grad):
        x = self.x
        return grad * self.lmbda * np.where(x > 0, 1, self.alpha * np.exp(x)).astype(x.dtype)


class GELU(Activation):

    def forward(self, x):
        self.x = x
        return (
                0.5
                * x
                * (
                    1
                    + np.tanh(
                        np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
                    )
                )
            )

    def backward(self, grad):
        x = self.x
        sech = lambda z: 2 / (np.exp(z) + np.exp(-z))

        return grad *(
            0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x)
            + (0.0535161 * np.power(x, 3) + 0.398942 * x)
            * np.power(sech(0.0356774 * np.power(x, 3) + 0.797885 * x), 2)
            + 0.5
        ).astype(x.dtype)

class Identity(Activation):

    def forward(self, x):
        self.x = x
        return x

    def backward(self, grad):
        x = self.x
        return np.asarray(grad * np.ones(x.shape).astype(x.dtype))

    
activations= {
    
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "softmax": Softmax(),
    "softplus": Softplus(),
    "softsign": Softsign(),
    "swish": Swish(),
    "mish": Mish(),
    "tanh_exp": TanhExp(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "elu": ELU(),
    "selu": SELU(),
    "gelu": GELU(),
    None: Identity()
    
}