import numpy as np
from transformer.layers.base.dropout import Dropout


class PositionalEncoding():
    """ Implements the sinusoidal positional encoding.
    """

    def __init__(self,max_len, d_model, dropout_rate=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = Dropout(dropout_rate)
 
        pe = np.zeros((max_len, d_model))  # (max_len, d_model)
        position = np.arange(0, max_len)[:, np.newaxis]# (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model,)

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[:, np.newaxis, :]  # (max_len, 1, d_model)


    def forward(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.shape[0], :]  # (batch_size, seq_len, d_model)
        x = self.dropout.forward(x)  # (batch_size, seq_len, d_model)

        # x: (batch_size, seq_len, d_model)
        return x

    def backward(self, error):
        """ error: (batch_size, seq_len, d_model)
        """
        error = self.dropout.backward(error)

        return error

