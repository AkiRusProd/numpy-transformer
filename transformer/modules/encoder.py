import numpy as np
from transformer.layers.base.embedding import Embedding
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.encoder_layer import EncoderLayer
from transformer.layers.combined.positional_encoding import PositionalEncoding


class Encoder:
    def __init__(self, src_vocab_size, heads_num, layers_num, d_model, d_ff, dropout, max_length = 100):
        # super(Encoder, self).__init__()

        self.token_embedding    = Embedding(src_vocab_size, d_model, max_length)
        self.position_embedding = PositionalEncoding(max_length, d_model, dropout) #pos embeding

        self.layers = []
        for _ in range(layers_num):
            self.layers.append(EncoderLayer(d_model, heads_num, d_ff, dropout))

        self.dropout = Dropout(dropout)
        self.scale = np.sqrt(d_model)

    def forward(self, src, src_mask):
        # batchsize, seq_length = src.shape
        # positions = np.tile(np.arange(0, seq_length), (batchsize, 1))
        # src = self.dropout.forward((self.token_embedding.forward(src) * self.scale + self.position_embedding.forward(positions)))
        src = self.token_embedding.forward(src)
        src = self.position_embedding.forward(src)
        src = self.dropout.forward(src)

        for layer in self.layers:
            src = layer.forward(src, src_mask)

        return src

# import torch
# # positions = torch.arange(0, 10).unsqueeze(0).repeat(30, 1)
# positions = torch.arange(0, 10).expand(30, 10)
# print(positions.shape)
# positions = np.tile(np.arange(0, 10), (30, 1))

# print(positions.shape)