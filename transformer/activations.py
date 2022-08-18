import numpy as np


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
        f_x = self.forward(self.x)

        return grad * f_x * (1.0 - f_x)


class Tanh(Activation):

    def forward(self, x):
        self.x = x

        return np.tanh(x)

    def backward(self, grad):
        x = self.x
        return grad * (1.0 - np.power(self.forward(x), 2))


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
        
        J = softmax[..., np.newaxis] * np.tile(np.identity(softmax.shape[-1]), (softmax.shape[0], *np.ones(softmax.ndim, dtype = np.int8))) - (softmax[..., np.newaxis, :].transpose(*np.arange(0, softmax.ndim - 1, 1, dtype=np.int8), -1, -2) @ softmax[..., np.newaxis, :]) #np.matmul(softmax[:, :, None], softmax[:, None, :])
        input_grad =  grad[..., np.newaxis, :] @ J
        
        return input_grad.reshape(self.x.shape) / batch_size


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

    def jacobian_backward(self, grad = None):
        batch_size = self.x.shape[0]
        softmax = self.softmax_forward(self.x)

        J = np.tile(np.identity(softmax.shape[-1]), (softmax.shape[0], *np.ones(softmax.ndim, dtype = np.int8))) - (np.ones((*self.x.shape, 1)) @ softmax[..., np.newaxis, :])
        
        input_grad =  grad[..., np.newaxis, :] @ J
        
        return input_grad.reshape(self.x.shape) / batch_size


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
        return grad * 1 / np.power(1 + np.abs(x), 2)

class Swish(Activation):

    def __init__(self, beta = 1):
        self.beta = beta

    def forward(self, x):
        self.x = x
        self.sigmoid = lambda z: 1 / (1 + np.exp(-z)) 

        return x * self.sigmoid(self.beta * x)

    def backward(self, grad):
        f_x = self.forward(self.x)

        return grad * (self.beta * f_x + self.sigmoid(self.beta * self.x) * (1 - self.beta * f_x))

class Mish(Activation):

    def forward(self, x):
        self.x = x
        return x * np.tanh(np.log(1 + np.exp(x)))

    def backward(self, grad):
        x = self.x
        return grad * (np.exp(x) * (4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)) / np.power((2 * np.exp(x) + np.exp(2 * x) + 2), 2))

class TanhExp(Activation):

    def forward(self, x):
        self.x = x

        return x * np.tanh(np.exp(x))

    def backward(self, grad):
        x = self.x
        return grad * (np.tanh(np.exp(x)) - x * np.exp(x) * (np.power(np.tanh(np.exp(x)), 2) - 1))


class ReLU(Activation):

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, 0, 1)


class LeakyReLU(Activation):

    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x <= 0, self.alpha * x, x)

    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, self.alpha, 1)


class ELU(Activation):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha 

    def forward(self, x):
        self.x = x
        return np.where(x <= 0, self.alpha * (np.exp(x) - 1), x)

    def backward(self, grad):
        x = self.x
        return grad * np.where(x <= 0, self.alpha + self.forward(x), 1)


class SELU(Activation):

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.lmbda = 1.0507009873554804934193349852946 

    def forward(self, x):
        self.x = x
        return self.lmbda * np.where(x > 0, x, self.alpha*(np.exp(x)-1))

    def backward(self, grad):
        x = self.x
        return grad * self.lmbda * np.where(x > 0, 1, self.alpha * np.exp(x))


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
        )

class Identity(Activation):

    def forward(self, x):
        self.x = x
        return x

    def backward(self, grad):
        x = self.x
        return grad * np.ones(x.shape)

    
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