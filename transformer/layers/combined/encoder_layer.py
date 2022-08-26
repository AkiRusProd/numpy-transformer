from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.self_attention import MultiHeadAttention
from transformer.layers.combined.positionwise_feed_forward import PositionwiseFeedforward
from transformer.layers.base.layer_norm import LayerNormalization


class EncoderLayer:
    def __init__(self, d_model, heads_num, d_ff, dropout, data_type):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = LayerNormalization(d_model, epsilon=1e-6, data_type=data_type)
        self.ff_layer_norm       = LayerNormalization(d_model, epsilon=1e-6, data_type=data_type)
        self.self_attention = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.position_wise_feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout, data_type)

    def forward(self, src, src_mask, training):
        _src, _ = self.self_attention.forward(src, src, src, src_mask, training)
        src = self.self_attention_norm.forward(src + self.dropout.forward(_src, training))
        
        _src = self.position_wise_feed_forward.forward(src, training)
        src = self.ff_layer_norm.forward(src + self.dropout.forward(_src, training))

        return src

    def backward(self, error):
        error = self.ff_layer_norm.backward(error)

        _error = self.position_wise_feed_forward.backward(self.dropout.backward(error))
        error = self.self_attention_norm.backward(error + _error)
        
        _error, _error2, _error3 = self.self_attention.backward(self.dropout.backward(error))

        return _error +_error2 +_error3 + error

    def set_optimizer(self, optimizer):
        self.self_attention_norm.set_optimizer(optimizer)
        self.ff_layer_norm.set_optimizer(optimizer)
        self.self_attention.set_optimizer(optimizer)
        self.position_wise_feed_forward.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.self_attention_norm.update_weights(layer_num)
        layer_num = self.ff_layer_norm.update_weights(layer_num)
        layer_num = self.self_attention.update_weights(layer_num)
        layer_num = self.position_wise_feed_forward.update_weights(layer_num)

        return layer_num