import numpy as np


class Dropout():
    """
    Applies dropout to the input data
    ---------------------------------
        Args:
            `rate` (float): the rate from `"0.0 to 1.0"` of dropout
        Returns:
            output: the dropout input data with same shape
    """

    def __init__(self, rate = 0.1) -> None:
        self.rate = rate
        self.input_shape = None

    def build(self):
        self.output_shape = self.input_shape

    def forward(self, X, training = True):
        self.mask = np.random.binomial(
                        n = 1,
                        p = 1 - self.rate,
                        size = X.shape,
                    )

        return X * self.mask

    def backward(self, error):

        return error * self.mask