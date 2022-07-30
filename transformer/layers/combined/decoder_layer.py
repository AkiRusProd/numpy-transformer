import numpy as np
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.self_attention import MultiHeadAttention
from transformer.layers.combined.positionwise_feed_forward import PositionwiseFeedforward
from transformer.layers.base.layer_norm import LayerNormalization

class DecoderLayer():
    def __init__(self, d_model, heads_num, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = LayerNormalization(d_model)
        self.enc_attn_layer_norm = LayerNormalization(d_model)
        self.ff_layer_norm = LayerNormalization(d_model)
        self.self_attention = MultiHeadAttention(d_model, heads_num, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, heads_num, dropout)
        self.position_wise_feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout)

    def forward(self, trg, trg_mask, src, src_mask):
        _trg, _ = self.self_attention.forward(trg, trg, trg, trg_mask)
        trg = self.self_attention_norm.forward(trg + self.dropout.forward(_trg))

        _trg, attention = self.encoder_attention.forward(trg, src, src, src_mask)
        trg = self.enc_attn_layer_norm.forward(trg + self.dropout.forward(_trg))


        _trg = self.position_wise_feed_forward.forward(trg)
        trg = self.ff_layer_norm.forward(trg + self.dropout.forward(_trg))

        return trg, attention