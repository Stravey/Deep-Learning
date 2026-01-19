# RNN
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

import TP

batch_size,num_steps = 32,35
train_iter,vocab = TP.load_data_time_machine(batch_size, num_steps)

# 独热编码 one-hot
print(F.one_hot(torch.tensor([0,2]),len(vocab)))
X = torch.arange(10).reshape((2,5))
print(F.one_hot(X.T,28).shape)

# 初始化参数模型
def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size = shape,device = device) * 0.01

    W_xh = normal((num_inputs,num_hiddens))
    W_hh = normal((num_hiddens,num_hiddens))
    b_h = torch.zeros(num_hiddens,device=device)
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 循环神经网络模型

