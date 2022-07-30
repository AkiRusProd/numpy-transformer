import numpy as np
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.self_attention import MultiHeadAttention
from transformer.layers.combined.positionwise_feed_forward import PositionwiseFeedforward
from transformer.layers.base.layer_norm import LayerNormalization


class EncoderLayer:
    def __init__(self, d_model, heads_num, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = LayerNormalization(d_model)
        self.ff_layer_norm = LayerNormalization(d_model)
        self.self_attention = MultiHeadAttention(d_model, heads_num, dropout)
        self.position_wise_feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention.forward(src, src, src, src_mask)
        src = self.self_attention_norm.forward(src + self.dropout.forward(_src))
        
        _src = self.position_wise_feed_forward.forward(src)
        src = self.ff_layer_norm.forward(src + self.dropout.forward(_src))

        return src