# Bahdanau注意力
import torch
from torch import nn
from d2l import torch as d2l

#@save
class AttentionDecoder(nn.Module):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError



