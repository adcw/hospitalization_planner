import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LazyLinear

from src.nn.archs.lazy_fccn import LazyFCCN

"""
https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch
"""


class LazySelfAttention(nn.Module):
    def __init__(self):
        super(LazySelfAttention, self).__init__()
        self.in_features = None

        # Linear transformations for Q, K, V from the same source
        self.key = None  # LazyFCCN()  # Adding dummy size, it will be changed anyway
        self.query = None  # LazyLinear(4)  # as above
        self.value = None  # LazyLinear(4)  # as above

    def forward(self, x: torch.Tensor, mask=None):

        # Infer feature size if not initialized
        if self.in_features is None:
            self.in_features = x.size(-1)

            self.key = LazyFCCN([self.in_features, self.in_features], self.in_features, device=x.device,
                                activation=F.selu, dropout_rate=0.16)
            self.query = LazyFCCN([self.in_features, self.in_features], self.in_features, device=x.device,
                                  activation=F.selu, dropout_rate=0.16)
            self.value = LazyFCCN([self.in_features, self.in_features], self.in_features, device=x.device,
                                  activation=F.selu, dropout_rate=0.16)

            # self.key.out_features = self.in_features
            # self.query.out_features = self.in_features
            # self.value.out_features = self.in_features
            #
            # self.init_layer(self.key, x)
            # self.init_layer(self.query, x)
            # self.init_layer(self.value, x)

        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = F.scaled_dot_product_attention(queries, keys, values)

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = attention_weights * values

        return output, attention_weights

    @staticmethod
    def init_layer(layer, input_):
        with torch.no_grad():
            layer.in_features = input_.shape[-1]
            layer.weight.materialize((layer.out_features, layer.in_features))
            if layer.bias is not None:
                layer.bias.materialize((layer.out_features,))
            layer.reset_parameters()
