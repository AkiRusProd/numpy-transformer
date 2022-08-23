import numpy as np


class Embedding():
    """
    Add Embedding layer
    ---------------
        Args:
            `input_dim`: (int), size of vocabulary
            `output_dim` (int): number of neurons in the layer (vector size)
        Returns:
            input: data with shape (batch_size, input_length)
            output: data with shape (batch_size, input_length, output_dim)
    """

    def __init__(self, input_dim, output_dim, data_type = np.float32):
        self.input_dim = input_dim
        self.output_dim   = output_dim

        self.w = None

        self.optimizer = None
        self.data_type = data_type

        self.build()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        
        self.w = np.random.normal(0, pow(self.input_dim, -0.5), (self.input_dim, self.output_dim)).astype(self.data_type)

        self.v, self.m         = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)
        self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)

        # if self.input_length is not None:
        #     self.output_shape = (self.input_length, self.output_dim)
        # else:
        #     print("Input_length is not set, can`t compute output_shape of the Embedding layer")

    #one hot encoding
    def prepare_labels(self, batch_labels):
        batch_labels = batch_labels.astype(np.int32)
        
        prepared_batch_labels = np.zeros((batch_labels.size,  self.input_dim)) #batch_labels.max() + 1)
        prepared_batch_labels[np.arange(batch_labels.size), batch_labels.reshape(1, -1)] = 1

        return prepared_batch_labels.reshape(self.batch_size, self.current_input_length, self.input_dim).astype(self.data_type) 


    def forward(self, X):
        self.input_data = X # (batch_size, input_length); inputs: values of vocabulary from 0 to input_dim - 1
        
        if not all([np.array_equal(len(self.input_data[0]), len(arr)) for arr in self.input_data]):
            raise ValueError("Input sequences must be of the same length")

        self.current_input_length = len(self.input_data[0])
        self.batch_size = len(self.input_data)

        self.input_data = self.prepare_labels(self.input_data)

        self.output_data = np.dot(self.input_data, self.w)
        
        return self.output_data

    def backward(self, error):        
        # self.grad_w = np.dot(np.transpose(self.input_data, axes = (0, 2, 1)), error).sum(axis = 0).sum(axis = 1).reshape(self.w.shape)
        self.grad_w = np.matmul(np.transpose(self.input_data, axes = (0, 2, 1)), error).sum(axis = 0)

        # output_error = np.dot(error, self.w.T)

        # return output_error
        return None

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)

        return layer_num + 1
    def get_grads(self):
        return self.grad_w, self.grad_b

    def set_grads(self, grads):
        self.grad_w, self.grad_b = grads
        
