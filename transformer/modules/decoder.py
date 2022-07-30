import numpy as np
from transformer.layers.base.activation import Activation
from transformer.layers.base.dense import Dense
from transformer.layers.base.embedding import Embedding
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.decoder_layer import DecoderLayer
from transformer.activations import Softmax
from transformer.layers.combined.positional_encoding import PositionalEncoding

class Decoder:
    def __init__(self, trg_vocab_size, heads_num, layers_num, d_model, d_ff, dropout, max_length = 100):
        # super(Decoder, self).__init__()

        self.token_embedding    = Embedding(trg_vocab_size, d_model, max_length)
        self.position_embedding = PositionalEncoding(max_length, d_model, dropout) #pos embeding

        self.layers = []
        for _ in range(layers_num):
            self.layers.append(DecoderLayer(d_model, heads_num, d_ff, dropout))

        self.fc_out = Dense(input_shape = d_model, units_num = trg_vocab_size)
        self.dropout = Dropout(dropout)
        self.scale = np.sqrt(d_model)

        self.activation = Activation(Softmax())


    def forward(self, trg, trg_mask, src, src_mask):
        batchsize, seq_length = trg.shape
        # positions = np.tile(np.arange(0, seq_length), (batchsize, 1))
        # trg = self.dropout.forward((self.token_embedding.forward(trg) * self.scale + self.position_embedding.forward(positions)))
        trg = self.token_embedding.forward(trg)
        trg = self.position_embedding.forward(trg)
        trg = self.dropout.forward(trg)
        # print(f"{trg.shape=}, {trg_mask.shape=}, {src.shape=}, {src_mask.shape=}")
        for layer in self.layers:
            trg, attention = layer.forward(trg, trg_mask, src, src_mask)
        # print(trg.shape)
        output = self.fc_out.forward(trg)

        activated_output = self.activation.forward(output)


        return activated_output, attention