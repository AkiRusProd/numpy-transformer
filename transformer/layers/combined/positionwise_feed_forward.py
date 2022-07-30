import numpy as np
from transformer.activations import ReLU
from transformer.layers.base.activation import Activation
from transformer.layers.base.dense import Dense
from transformer.layers.base.dropout import Dropout

class PositionwiseFeedforward():
    def __init__(self, d_model = 512, d_ff = 2048,  dropout = 0.1):
        
        self.fc_1 = Dense(input_shape = d_model, units_num = d_ff, use_bias = False)
        self.activation = Activation(ReLU())
        self.fc_2 = Dense(input_shape = d_ff, units_num = d_model, use_bias = False)

        self.dropout = Dropout(dropout)

    def forward(self, X):
        
        X = self.fc_1.forward(X)
        X = self.activation.forward(X)
        X = self.dropout.forward(X)
        X = self.fc_2.forward(X)
    
        return X
