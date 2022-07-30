import numpy as np
from transformer.layers.base.dense import Dense
from transformer.layers.base.dropout import Dropout
from transformer.activations import Softmax


class MultiHeadAttention:
    """Multi-HeadAttention"""
    def __init__(self, d_model = 512, heads_num = 8, dropout = 0.1):
        self.d_model = d_model
        self.heads_num = heads_num
        self.dropout = dropout

        self.d_k, self.d_q, self.d_v = self.d_model // heads_num, self.d_model // heads_num, self.d_model // heads_num #512 / 8 = 64
        
        self.K_linear = Dense(input_shape = self.d_model, units_num = self.d_k * heads_num, use_bias = False) # self.W_K = np.random.randn(self.d_model, self.d_k)
        self.Q_linear = Dense(input_shape = self.d_model, units_num = self.d_q * heads_num, use_bias = False) # self.W_Q = np.random.randn(self.d_model, self.d_q)
        self.V_linear = Dense(input_shape = self.d_model, units_num = self.d_v * heads_num, use_bias = False) # self.W_V = np.random.randn(self.d_model, self.d_v)
        self.O_linear = Dense(input_shape = self.d_model, units_num = self.d_v * heads_num) # self.W_O = np.random.randn(self.d_model, self.heads_num * self.d_v)

        self.activation = Softmax()

        self.dropout = Dropout(dropout)


    def forward(self, query, key, value, mask):
        batch_size = key.shape[0]
        key_len, query_len, value_len = key.shape[1], query.shape[1], value.shape[1]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        K = self.K_linear.forward(key)
        Q = self.Q_linear.forward(query)
        V = self.V_linear.forward(value)

        # split heads_num
        K = K.reshape(batch_size, self.heads_num, key_len, self.d_k)
        Q = Q.reshape(batch_size, self.heads_num, query_len, self.d_q)
        V = V.reshape(batch_size, self.heads_num, value_len, self.d_v)

        # print(Q.shape, K.shape)
        energy = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            mask = mask[:, np.newaxis, ...]
            # print(energy.shape, mask.shape, query_len, key_len)
            energy = np.where(mask == 0, float("-1e20"), energy)
        # print(energy.shape)
        attention = self.activation.function(energy)

        output = np.matmul(self.dropout.forward(attention), V)
        # print(output.shape)
        concat_output = output.reshape(batch_size, query_len, self.heads_num * self.d_v) #self.d_model

        O = self.O_linear.forward(concat_output)

        return O, attention


        

        