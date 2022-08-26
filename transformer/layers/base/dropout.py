try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False



class Dropout():
    """
    Applies dropout to the input data
    ---------------------------------
        Args:
            `rate` (float): the rate from `"0.0 to 1.0"` of dropout
        Returns:
            output: the dropout input data with same shape
    """

    def __init__(self, rate = 0.1, data_type = np.float32) -> None:
        self.rate = rate
        self.input_shape = None

        self.data_type = data_type

    def build(self):
        self.output_shape = self.input_shape

    def forward(self, X, training = True):

        self.mask = 1.0
        if training: self.mask = np.random.binomial(
                        n = 1,
                        p = 1 - self.rate,
                        size = X.shape,
                    ).astype(self.data_type)

        return X * self.mask

    def backward(self, error):

        return error * self.mask