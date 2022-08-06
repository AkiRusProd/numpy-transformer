# import sys
# from pathlib import Path
# sys.path[0] = str(Path(sys.path[0]).parent)

# import numpy as np
# from transformer.layers.base.dense import Dense
# from transformer.layers.base.activation import Activation
# from transformer.layers.base.dropout import Dropout
# from transformer.activations import Softmax


# class MultiHeadAttention:
#     """Multi-HeadAttention"""
#     def __init__(self, d_model = 512, heads_num = 8, dropout = 0):
#         self.d_model = d_model
#         self.heads_num = heads_num
#         self.dropout = dropout

#         self.d_k, self.d_q, self.d_v = self.d_model // heads_num, self.d_model // heads_num, self.d_model // heads_num #512 / 8 = 64
        
#         self.K_linear = Dense(input_shape = self.d_model, units_num = self.d_k * heads_num, use_bias = False) # self.W_K = np.random.randn(self.d_model, self.d_k)
#         self.Q_linear = Dense(input_shape = self.d_model, units_num = self.d_q * heads_num, use_bias = False) # self.W_Q = np.random.randn(self.d_model, self.d_q)
#         self.V_linear = Dense(input_shape = self.d_model, units_num = self.d_v * heads_num, use_bias = False) # self.W_V = np.random.randn(self.d_model, self.d_v)
#         self.O_linear = Dense(input_shape = self.d_model, units_num = self.d_v * heads_num, use_bias=False) # self.W_O = np.random.randn(self.d_model, self.heads_num * self.d_v)

#         self.activation = Activation(Softmax())

#         self.dropout = Dropout(dropout)

#     def split_heads_forward(self, x):
#         """Split the last dimension into (num_heads, depth).
#         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#         """
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]
#         depth = x.shape[2]

#         return x.reshape(batch_size, -1, self.heads_num, self.d_k).transpose(0, 2, 1, 3)

#     def split_heads_backward(self, x):
#         """Split the last dimension into (num_heads, depth).
#         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#         """
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]
#         depth = x.shape[2]
#         #x.transpose(0, 2, 1, 3).reshape(batch_size, self.key_len, self.d_model)
#         return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * self.d_k)

#     def group_heads_forward(self, x):
#         """Group the last dimension into (num_heads, depth).
#         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#         """
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]
#         depth = x.shape[2]

#         return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * self.d_k)

#     def group_heads_backward(self, x):
#         """Group the last dimension into (num_heads, depth).
#         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#         """
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]
#         depth = x.shape[2]

#         return x.reshape(batch_size, -1, self.heads_num, self.d_k).transpose(0, 2, 1, 3)
        


#     def forward(self, query, key, value, mask):
#         batch_size = key.shape[0]
        
#         self.key_len, self.query_len, self.value_len = key.shape[1], query.shape[1], value.shape[1]

#         #query = [batch size, query len, hid dim]
#         #key = [batch size, key len, hid dim]
#         #value = [batch size, value len, hid dim]

#         K = self.K_linear.forward(key)
#         Q = self.Q_linear.forward(query)
#         V = self.V_linear.forward(value)
#         # print("K", K )

#         # print(K.shape, Q.shape, V.shape)
#         # # split heads_num not equal splits
#         # self.K = K.reshape(batch_size, self.heads_num, self.key_len, self.d_k)
#         # self.Q = Q.reshape(batch_size, self.heads_num, self.query_len, self.d_q)
#         # self.V = V.reshape(batch_size, self.heads_num, self.value_len, self.d_v)
#         self.K = self.split_heads_forward(K)
#         self.Q = self.split_heads_forward(Q)
#         self.V = self.split_heads_forward(V)
#         # print(self.K[0])
#         # print("Q", Q)
#         # print("V", V)

#         # print(Q.shape, K.shape)
#         energy = np.matmul(self.Q, self.K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
#         # print("energy", energy.shape)
#         self.mask = mask
#         if self.mask is not None:
#             self.mask = self.mask[:, np.newaxis, ...]
#             # print(energy.shape, mask.shape, query_len, key_len)
#             energy = np.where(self.mask == 0, float('-inf'), energy)#float("-1e20")
#         # print(energy.shape)
#         attention = self.activation.forward(energy)
        
#         self.dropout_attention = self.dropout.forward(attention)
#         output = np.matmul(self.dropout_attention, self.V)
#         # print("attention", attention.shape, attention[0][0])
#         # concat_output = output.reshape(batch_size, self.query_len, self.heads_num * self.d_v) #self.d_model
#         concat_output = self.group_heads_forward(output)
#         print(np.equal(output, self.group_heads_backward(concat_output)))
#         # print("group_heads", concat_output.shape, concat_output[0][0])
#         # self.debug_group_heads = concat_output

#         O = self.O_linear.forward(concat_output.astype(np.float32))
#         print("O", O.shape, O)
#         return O#, attention

#     def backward(self, error):
#         error = self.O_linear.backward(error)
        
#         # error = error.reshape(error.shape[0], self.heads_num, self.query_len, self.d_v)
#         error = self.group_heads_backward(error)
#         V_error = np.matmul(self.dropout_attention.transpose(0, 1, 3, 2), error)
#         # V_error = np.matmul(error.transpose(0, 1, 3, 2), self.dropout_attention) #alter
#         error = np.matmul(error, self.V.transpose(0, 1, 3, 2))
#         error = self.dropout.backward(error)
#         error = self.activation.backward(error)

#         if self.mask is not None:
#             error = np.where(self.mask == 0, 0, error)

#         error /= np.sqrt(self.d_k)

#         Q_error = np.matmul(error, self.K)
#         # K_error = np.matmul(error.transpose(0, 1, 3, 2), self.Q)
#         K_error = np.matmul(self.Q.transpose(0, 1, 3, 2), error) #alter
#         K_error = K_error.transpose(0, 1, 3, 2)

        
#         # V_error = V_error.reshape(V_error.shape[0], self.value_len, self.d_model)
#         # Q_error = Q_error.reshape(Q_error.shape[0], self.query_len, self.d_model)
#         # K_error = K_error.reshape(K_error.shape[0], self.key_len, self.d_model)
#         V_error = self.split_heads_backward(V_error)
#         Q_error = self.split_heads_backward(Q_error)
#         K_error = self.split_heads_backward(K_error)
#         # print(np.equal(V_error, V_error_alter).all())



#         V_error = self.V_linear.backward(V_error)
#         Q_error = self.Q_linear.backward(Q_error)
#         K_error = self.K_linear.backward(K_error)

#         return Q_error, K_error, V_error


        

        

#     def set_optimizer(self, optimizer):
#         self.K_linear.set_optimizer(optimizer)
#         self.Q_linear.set_optimizer(optimizer)
#         self.V_linear.set_optimizer(optimizer)
#         self.O_linear.set_optimizer(optimizer)

#     def update_weights(self, layer_num):
#         layer_num = self.K_linear.update_weights(layer_num)
#         layer_num = self.Q_linear.update_weights(layer_num)
#         layer_num = self.V_linear.update_weights(layer_num)
#         layer_num = self.O_linear.update_weights(layer_num)

#         return layer_num


# model = MultiHeadAttention()

# K_w = np.random.normal(0, 1, (model.d_model, model.d_model)).astype('float32')
# Q_w = np.random.normal(0, 1, (model.d_model, model.d_model)).astype('float32')
# V_w = np.random.normal(0, 1, (model.d_model, model.d_model)).astype('float32')
# O_w = np.random.normal(0, 1, (model.d_model, model.d_model)).astype('float32')

# model.K_linear.w = K_w
# model.Q_linear.w = Q_w
# model.V_linear.w = V_w
# model.O_linear.w = O_w

# input = np.random.normal(0, 1, (2, 10, model.d_model)).astype('float32')

# output = model.forward(input, input, input, mask = None)

# # print("output shape:", output.shape)
# # print("output:", output)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ScaledDotProductAttention(nn.Module):
#     """ Computes scaled dot product attention
#     """

#     def __init__(self, scale, dropout_rate):
#         super(ScaledDotProductAttention, self).__init__()
#         self.scale = scale
#         self.dropout_rate = dropout_rate
        
#         self.dropout = nn.Dropout(dropout_rate)


#     def forward(self, query, key, value, mask=None):
#         """ query: (batch_size, n_heads, query_len, head_dim)
#             key: (batch_size, n_heads, key_len, head_dim)
#             value: (batch_size, n_heads, value_len, head_dim)
#             mask: (batch_size, 1, 1, source_seq_len) for source mask
#                   (batch_size, 1, target_seq_len, target_seq_len) for target mask
#         """
#         # calculate alignment scores
#         scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, n_heads, query_len, value_len)
#         scores = scores / self.scale  # (batch_size, num_heads, query_len, value_len)
#         # print("scores", scores.shape)
#         # mask out invalid positions
#         # print(key.shape, query.shape, value.shape, mask.shape)
#         if mask is not None:
#             # print(scores.shape, mask.shape)
#             scores = scores.masked_fill(mask == 0, float('-inf'))  # (batch_size, n_heads, query_len, value_len)
#         # print(scores)
#         # print(scores.shape)
#         # calculate the attention weights (prob) from alignment scores
#         attn_probs = F.softmax(scores, dim=-1)  # (batch_size, n_heads, query_len, value_len)
#         # print("attention", attn_probs.shape, attn_probs[0][0])
#         # calculate context vector
#         output = torch.matmul(self.dropout(attn_probs), value)  # (batch_size, n_heads, query_len, head_dim)
#         # print("attention", output)
#         # output: (batch_size, n_heads, query_len, head_dim)
#         # attn_probs: (batch_size, n_heads, query_len, value_len)
#         return output, attn_probs




# class MultiHeadAttention(nn.Module):
#     """ Implements Multi-Head Self-Attention proposed by Vaswani et al., 2017.
#         refer https://arxiv.org/abs/1706.03762
#     """

#     def __init__(self, d_model, n_heads, dropout_rate=0.1):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % n_heads == 0, "`d_model` should be a multiple of `n_heads`"

#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_k = self.d_v = d_model // n_heads  # head_dim
#         self.dropout_rate = dropout_rate

#         self.W_q = nn.Linear(d_model, d_model, bias=False)
#         self.W_k = nn.Linear(d_model, d_model, bias=False)
#         self.W_v = nn.Linear(d_model, d_model, bias=False)
#         self.W_o = nn.Linear(d_model, d_model, bias=False)

#         self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)
    

#     def split_heads(self, x, seq_len):
#         """ x: (batch_size, seq_len, d_model)
#         """
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
#         # x = torch.reshape(x, (batch_size, self.n_heads, seq_len, self.d_k))

#         # x: (batch_size, n_heads, seq_len, head_dim)
#         return x


#     def group_heads(self, x, seq_len):
#         """ x: (batch_size, n_heads, seq_len, head_dim)
#         """
#         batch_size = x.size(0)
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
#         # x = torch.reshape(x, (batch_size, seq_len, self.d_model))

#         # x: (batch_size, seq_len, d_model)
#         return x


#     def forward(self, query, key, value, mask=None):
#         """ query: (batch_size, query_len, d_model)
#             key: (batch_size, key_len, d_model)
#             value: (batch_size, value_len, d_model)
#             mask: (batch_size, 1, source_seq_len) for source mask
#                   (batch_size, target_seq_len, target_seq_len) for target mask
#         """
#         self.key_len, self.query_len, self.value_len = key.shape[1], query.shape[1], value.shape[1]

#         # apply linear projections to query, key and value
#         Q = self.split_heads(self.W_q(query), self.query_len)  # (batch_size, n_heads, query_len, head_dim)
#         K = self.split_heads(self.W_k(key), self.key_len)  # (batch_size, n_heads, key_len, head_dim)
#         V = self.split_heads(self.W_v(value), self.value_len)  # (batch_size, n_heads, value_len, head_dim)
        
#         # print("K", K[0])
#         # print("Q", Q)
#         # print("V", V)

#         if mask is not None:
#             # apply same mask for all the heads
#             mask = mask.unsqueeze(1)

#             # mask: (batch_size, 1, 1, source_seq_len) for source mask
#             #       (batch_size, 1, target_seq_len, target_seq_len) for target mask
        
#         # calculate attention weights and context vector for each of the heads
#         x, attn = self.attention(Q, K, V, mask)

#         # x: (batch_size, n_heads, query_len, head_dim)
#         # attn: (batch_size, n_heads, query_len, value_len)

#         # concatenate context vector of all the heads
#         x = self.group_heads(x, self.query_len)  # (batch_size, query_len, d_model)
#         # print("group_heads", x.shape, x[0][0])
#         # self.debug_group_heads = x
#         # apply linear projection to concatenated context vector
#         x = self.W_o(x)  # (batch_size, query_len, d_model)
#         print("x", x.shape, x)
#         # x: (batch_size, query_len, d_model)
#         # attn: (batch_size, n_heads, query_len, value_len)
#         # print(x)
#         return x#, attn


# torch_model = MultiHeadAttention(d_model=512, n_heads=8, dropout_rate=0)


# with torch.no_grad():
#     torch_model.W_q.weight.copy_(torch.tensor(Q_w.T, requires_grad=True, dtype=torch.float32))
#     torch_model.W_k.weight.copy_(torch.tensor(K_w.T, requires_grad=True, dtype=torch.float32))
#     torch_model.W_v.weight.copy_(torch.tensor(V_w.T, requires_grad=True, dtype=torch.float32))
#     torch_model.W_o.weight.copy_(torch.tensor(O_w.T, requires_grad=True, dtype=torch.float32))


# torch_output = torch_model.forward(torch.tensor(input, dtype=torch.float32), 
#                     torch.tensor(input, dtype=torch.float32), 
#                     torch.tensor(input, dtype=torch.float32))


# criterion = nn.MSELoss()
# loss = criterion(torch_output, torch.tensor(output, dtype=torch.float32))
# loss.backward()
# print(loss)

#LINEAR: OK
#SPLIT HEADS: OK
#ENERGY: OK
#ATTENTION: OK

#GROUP HEADS: OK (make as in torch&&)
#OVERFLOW NUMPY EXP PROBLEM SOFTMAX: OK

# t_w = torch_model.W_o.weight.detach().numpy()
# w = model.O_linear.w.T
# print(np.equal(t_w, w).all()) #TRUE

# tgh = torch_model.debug_group_heads.detach().numpy()
# gh = model.debug_group_heads
# print(np.equal(tgh, gh).all()) #TRUE but false
# print(np.equal(torch_output.detach().numpy()[0], output[0])) #TRUE but false
# print(torch_output.detach().numpy())
# print(output.astype("float32"))
# print("torch_output shape:", torch_output.shape)
# print("torch_output: \n", torch_output)


# linear_output = model.O_linear.forward(output)
# print("linear_output shape:", linear_output)

# torch_linear_output = torch_model.W_o(torch_output)
# print("torch_linear_output shape:", torch_linear_output)

# torch_linear_output = torch_model.W_o(torch.tensor(output, dtype=torch.float32))
# print("torch_linear_output shape:", torch_linear_output)


# input = np.array([[1, 1]])

# dense = Dense(units_num = 2, input_shape = 2, use_bias=False)
# dense.w = np.random.normal(0, 1, (2, 2)).astype(np.float32)
# d_out = dense.forward(input)

# torch_dense = nn.Linear(2, 2, bias=False)
# with torch.no_grad():
#     torch_dense.weight.copy_(torch.tensor(dense.w.T, dtype=torch.float32, requires_grad=True))
# td_out = torch_dense(torch.tensor(input, dtype=torch.float32))

# print("d_out: \n", d_out)
# print("td_out: \n", td_out)

# grad = torch.autograd.grad(torch_model, torch_output, retain_graph=True)
# print("torch_model.W_q.grad: \n", torch_model.W_q.grad)


import numpy as np
# from nnmodel.exceptions.values_checker import ValuesChecker

class LayerNormalization():
    """
    Applies layer normalization to the input data
    ---------------------------------------------
        Args:
            `momentum` (float): the momentum parameter of the moving mean
            `epsilon` (float): the epsilon parameter of the algorithm
        Returns:
            output: the normalized input data with same shape
    """

    def __init__(self, epsilon = 0.001, input_shape = None):

        self.epsilon  = epsilon#ValuesChecker.check_float_variable(epsilon, "epsilon")

        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None

        self.moving_mean = None #Not using
        self.moving_var = None #Not using

        self.optimizer = None

        self.axis = None
        self.input_shape = input_shape#ValuesChecker.check_input_dim(input_shape, input_dim = None)

        self.build()
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    

    def build(self):
        # print(self.input_shape)
        # self.gamma = None#np.ones(self.input_shape)#(1)
        # self.beta = None#np.zeros(self.input_shape)#(1)


        # self.vg, self.mg         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        # self.vg_hat, self.mg_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        # self.vb, self.mb         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        # self.vb_hat, self.mb_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        

        self.output_shape = self.input_shape


    def forward(self, X):
        self.input_data = X
        self.batch_size = X.shape[0]

        if self.gamma is None:
            self.gamma = np.ones(self.input_data.shape[1:])
            self.beta = np.zeros(self.input_data.shape[1:])
    
            self.vg, self.mg         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
            self.vg_hat, self.mg_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

            self.vb, self.mb         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
            self.vb_hat, self.mb_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        # if self.axis is None: self.axis = tuple(np.arange(len(self.input_data.shape))[1:])
        
        # self.mean = np.mean(self.input_data, axis = self.axis, keepdims = True)
        # self.var = np.var(self.input_data, axis = self.axis, keepdims = True)
        
    
        # self.X_centered = (self.input_data - self.mean)
        # self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        # X_hat = self.X_centered * self.stddev_inv

        # self.output_data = self.gamma * X_hat + self.beta

        x_T = self.input_data.T

        self.mean = np.mean(x_T,axis = 0)
        self.var = np.var(x_T,axis = 0)

        self.X_centered = (x_T - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        # x_norm_T = (x_T - self.mean) / np.sqrt(self.var + self.epsilon)
        self.X_hat_T = self.X_centered * self.stddev_inv
        self.X_hat = self.X_hat_T.T
        self.output_data = self.gamma * self.X_hat + self.beta
        # cache = (x,x_norm,gamma,mean,var,eps)

        return self.output_data

        

    def backward(self, error, mode = "first"):
        # print(error.shape, error.shape[1:])
        # print(self.gamma.shape)
        # output_error = (1 / self.batch_size) * self.gamma * self.stddev_inv * (
        #     self.batch_size * error
        #     - np.sum(error, axis = self.axis, keepdims = True)
        #     - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(error * self.X_centered, axis = self.axis, keepdims = True)
        #     )

        # X_hat = self.X_centered * self.stddev_inv
        # self.grad_gamma = np.sum(error * X_hat)
        # self.grad_beta = np.sum(error)

        if mode == "first":
            x_T = self.input_data.T
            dout_T = error.T
            N = x_T.shape[0]
            # print(N)
            self.grad_beta = np.sum(error,axis = 0)
            self.grad_gamma = np.sum(self.X_hat * error,axis = 0)
            dx_norm = dout_T * self.gamma[np.newaxis, :].T#[np.newaxis,:]
            dv = ((x_T - self.mean) * -0.5 * (self.var + self.epsilon)** -1.5 * dx_norm).sum(axis = 0)
            dm = (dx_norm * -1 * (self.var + self.epsilon) ** -0.5).sum(axis = 0) + (dv * (x_T - self.mean) * -2 / N).sum(axis = 0)
            dx_T = dx_norm / (self.var + self.epsilon)** 0.5 + dv * 2 * (x_T - self.mean) / N + dm / N
            output_error = dx_T.T

        if mode == "second":
            dout_T = error.T
            # print((self.gamma[np.newaxis, :].T * self.stddev_inv).shape, self.gamma[np.newaxis, :].T.shape, self.stddev_inv.shape)
            # output_error = (1 / self.batch_size) * self.gamma[np.newaxis, :].T * self.stddev_inv * (
            #     self.batch_size * error.T
            #     - np.sum(error, axis = 0, keepdims = True)
            #     - self.X_centered * np.power(self.stddev_inv, 2) 
            #     * np.sum(error * self.X_centered.T, axis = 0, keepdims = True).T
            #     )
            x_T = self.input_data.T
            N = x_T.shape[0]

            dX_hat = dout_T * self.gamma[np.newaxis, :].T
            output_error = (1 / N) * self.stddev_inv * (
                N * dX_hat
                - np.sum(dX_hat, axis = 0)
                - self.X_hat_T * np.sum(dX_hat * self.X_hat_T, axis = 0)
            )
            output_error = output_error.T

        if mode == "third":
            dout_T = error.T
            x_T = self.input_data.T
            N = x_T.shape[0]
            dX_norm = dout_T * self.gamma[np.newaxis, :].T
            dvar = np.sum(dX_norm * self.X_centered, axis=0) * -.5 * self.stddev_inv**3
            dmu = np.sum(dX_norm * -self.stddev_inv, axis=0) + dvar * np.mean(-2. * self.X_centered, axis=0)

            output_error = (dX_norm * self.stddev_inv) + (dvar * 2 * self.X_centered / N) + (dmu / N)
            output_error = output_error.T

        if mode == "fourth":
            dout_T = error.T
            x_T = self.input_data.T
            N = x_T.shape[0]

            # output_error = (1 / N) * self.gamma[np.newaxis, :].T * self.stddev_inv * (
            #     N * error.T
            #     - np.sum(error.T, axis = 0)
            #     - self.X_centered * np.power(self.stddev_inv, 2) 
            #     * np.sum(error.T * self.X_centered, axis = 0)
            #     )
            x_T = self.input_data.T

            dX_norm = dout_T * self.gamma[np.newaxis, :].T
            dvar = np.sum(dX_norm * self.X_centered, axis=0) * -.5 * self.stddev_inv**3
            dmu = np.sum(dX_norm * -self.stddev_inv, axis=0) + dvar * np.mean(-2. * self.X_centered, axis=0)

            output_error = (dX_norm * self.stddev_inv) + (dvar * 2 * self.X_centered / N) + (dmu / N)
            output_error = output_error.T


        
        return output_error

    def update_weights(self, layer_num):
        # print(self.gamma.shape, self.beta.shape, self.grad_gamma.shape, self.grad_beta.shape)
        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat  = self.optimizer.update(self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, layer_num)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

        return layer_num + 1
        
    def get_grads(self):
        return self.grad_gamma, self.grad_beta

    def set_grads(self, grads):
        self.grad_gamma, self.grad_beta = grads


input = np.random.normal(0, 1, (2, 2, 10))
layer = LayerNormalization(input_shape = (2, 2, 10))
output = layer.forward(input)
output1 = layer.backward(output, mode = "first")
print("first")
print(output1)
output2 = layer.backward(output, mode = "second")
print("second")
print(output2)
output3 = layer.backward(output, mode = "third")
print("third")
print(output3)
output4 = layer.backward(output, mode = "fourth")
print("fourth")
print(output4)
print("-----------------------------------------------------")
print(output4.shape)
print("-----------------------------------------------------")

a = np.array([10, 3, 2])

print(np.prod(a[:-1]))