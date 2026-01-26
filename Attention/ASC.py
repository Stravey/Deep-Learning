# 注意力评分函数
import math
import torch
from torch import nn
from d2l import torch as d2l

#@save
# 掩蔽softmax dim=-1表示最后一个维度
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim = -1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value = -1e6)
        return nn.functional.softmax(X.reshape(shape), dim = -1)
