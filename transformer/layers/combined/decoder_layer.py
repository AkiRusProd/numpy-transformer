import numpy as np
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.self_attention import MultiHeadAttention
from transformer.layers.combined.positionwise_feed_forward import PositionwiseFeedforward
from transformer.layers.base.layer_norm import LayerNormalization

class DecoderLayer():
    def __init__(self, d_model, heads_num, d_ff, dropout, data_type):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = LayerNormalization(d_model, epsilon=1e-6, data_type = data_type)
        self.enc_attn_layer_norm = LayerNormalization(d_model, epsilon=1e-6, data_type = data_type)
        self.ff_layer_norm       = LayerNormalization(d_model, epsilon=1e-6, data_type = data_type)
        self.self_attention    = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.encoder_attention = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.position_wise_feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout, data_type)

    def forward(self, trg, trg_mask, src, src_mask):
        _trg, _ = self.self_attention.forward(trg, trg, trg, trg_mask)
        trg = self.self_attention_norm.forward(trg + self.dropout.forward(_trg))

        _trg, attention = self.encoder_attention.forward(trg, src, src, src_mask)
        trg = self.enc_attn_layer_norm.forward(trg + self.dropout.forward(_trg))


        _trg = self.position_wise_feed_forward.forward(trg)
        trg = self.ff_layer_norm.forward(trg + self.dropout.forward(_trg))

        return trg, attention

    def backward(self, error):
        error = self.ff_layer_norm.backward(error)

        _error = self.position_wise_feed_forward.backward(self.dropout.backward(error))
        error = self.enc_attn_layer_norm.backward(error + _error)

        _error, enc_error1, enc_error2 = self.encoder_attention.backward(self.dropout.backward(error))
        error = self.self_attention_norm.backward(error + _error)

        _error, _error2, _error3 = self.self_attention.backward(self.dropout.backward(error))
        
        return _error +_error2 + _error3 + error, enc_error1 + enc_error2

    def set_optimizer(self, optimizer):
        self.self_attention_norm.set_optimizer(optimizer)
        self.enc_attn_layer_norm.set_optimizer(optimizer)
        self.ff_layer_norm.set_optimizer(optimizer)
        self.self_attention.set_optimizer(optimizer)
        self.encoder_attention.set_optimizer(optimizer)
        self.position_wise_feed_forward.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.self_attention_norm.update_weights(layer_num)
        layer_num = self.enc_attn_layer_norm.update_weights(layer_num)
        layer_num = self.ff_layer_norm.update_weights(layer_num)
        layer_num = self.self_attention.update_weights(layer_num)
        layer_num = self.encoder_attention.update_weights(layer_num)
        layer_num = self.position_wise_feed_forward.update_weights(layer_num)

        return layer_num

