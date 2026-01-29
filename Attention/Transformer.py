# Transformer Architecture
import math
import torch
import pandas as pd
from torch import nn
from d2l import torch as d2l

# Position feed-forward network
class PositionWiseFFN(nn.Module):
    """The position-wise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
ffn = PositionWiseFFN(4, 8)
ffn.eval()
# print(ffn(torch.ones((2, 3, 4)))[0])


ln = nn.LayerNorm(2)
bn = nn.LazyBatchNorm1d()
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance from X in the training mode
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
