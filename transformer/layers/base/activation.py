import numpy as np
# from transformer.activations import activations
# from nnmodel.exceptions.values_checker import ValuesChecker

class Activation():
    """
    Applies the activation function to the input data
    --------------------------------------------------
        Args:
            `activation` (str) or (`ActivationFunction` class): activation function of the folowing:

        activations= {
            `"sigmoid": Sigmoid()`,
            `"tanh": Tanh()`,
            `"softmax": Softmax()`,
            `"softplus": Softplus()`,
            `"softsign": Softsign()`,
            `"relu": ReLU()`,
            `"leaky_relu": LeakyReLU()`,
            `"elu": ELU()`,
            `"selu": SELU()`,
            `"gelu": GELU()`,
            `None: Identity()`
        }

        Returns:
            output: the activated input data with same shape
    """

    def __init__(self, activation = None):
        self.input_shape = None

        self.activation = activation#ValuesChecker.check_activation(activation, activations)

    def build(self):
        self.output_shape = self.input_shape

    def forward(self, X, training = True):
        self.input_data = X
        return self.activation.function(X)

    def backward(self, error):
        # if self.activation.__class__.__name__ == "Softmax": #If act der is Jacobian matrix
        #     return self.activation.jacobian_derivative2(self.input_data, error)
        return error * self.activation.derivative(self.input_data)