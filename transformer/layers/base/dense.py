try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False



class Dense():
    """
    Add Dense layer
    ---------------
        Args:
            `units_num` (int): number of neurons in the layer
            `use_bias` (bool):  `True` if used. `False` if not used
        Returns:
            output: data with shape (batch_size, units_num)
    """

    def __init__(self, units_num, inputs_num = None, use_bias = True, data_type = np.float32):

        self.units_num   = units_num
        self.inputs_num = inputs_num
        self.use_bias    = use_bias
        
        self.w = None
        self.b = None

        self.optimizer = None
        self.data_type = data_type

        self.build()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):

        # self.w = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        #xavier initialization
        stdv = 1. / np.sqrt(self.inputs_num)# * 0.5 #input size
        # stdv = np.sqrt(6) / np.sqrt(self.input_size + self.units_num)
        #kaiming initialization
        # stdv = np.sqrt(2 / self.input_size)
        self.w = np.random.uniform(-stdv, stdv, (self.inputs_num, self.units_num)).astype(self.data_type)
        # if self.use_bias == True:
        #     self.b = np.random.uniform(-stdv, stdv, self.units_num)
        # else:
        #     self.b = np.zeros(self.units_num)
        self.b = np.zeros(self.units_num).astype(self.data_type)
        

        #glorot initialization
        # self.w = np.random.uniform(-np.sqrt(6 / (self.input_size + self.units_num)), np.sqrt(6 / (self.input_size + self.units_num)), (self.input_size, self.units_num))
        

        self.v, self.m         = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type) # optimizers params

        self.vb, self.mb         = np.zeros_like(self.b).astype(self.data_type), np.zeros_like(self.b).astype(self.data_type) # optimizers params
        self.vb_hat, self.mb_hat = np.zeros_like(self.b).astype(self.data_type), np.zeros_like(self.b).astype(self.data_type) # optimizers params

        self.output_shape = (1, self.units_num)

    def forward(self, X, training = True): 
        self.input_data = X
       
        self.batch_size = len(self.input_data)

        self.output_data = np.dot(self.input_data, self.w) + self.b
        
        return self.output_data

    def backward(self, error):
        self.grad_w = np.sum(np.matmul(self.input_data.transpose(0, 2, 1), error), axis = 0)
        self.grad_b = np.sum(error, axis = (0, 1))

        output_error = np.dot(error, self.w.T)

        return output_error

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
        if self.use_bias == True:
            self.b, self.vb, self.mb, self.vb_hat, self.mb_hat  = self.optimizer.update(self.grad_b, self.b, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)
        
        return layer_num + 1

    def get_grads(self):
        return self.grad_w, self.grad_b

    def set_grads(self, grads):
        self.grad_w, self.grad_b = grads
        
