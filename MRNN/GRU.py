import torch
from torch import nn
from d2l import torch as d2l

from RNN.RNNM import RNNModelScratch, train_model
from RNN.TP import load_data_time_machine

batch_size,num_steps = 32,35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens,device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size = shape,device = device) * 0.01

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device = device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device = device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hh, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 定义模型
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size,num_hiddens),device = device), )

# 定义gru模型
def gru(inputs,state,params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz + (H @ W_hz) + b_z))
        R = torch.sigmoid((X @ W_xr + (H @ W_hr) + b_r))
        H_tilda = torch.tanh((X@ W_xh + ((R * H) @ W_hh) + b_h))
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim = 0),(H,)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs,lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device,get_params,
                        init_gru_state, gru)

train_model(model,train_iter,vocab,lr,num_epochs,device)

# 训练结果

# 困惑度 1.0, 119219.5 词元/秒 cuda:0
# time travelleryou can show black is white by argument said filby
# travelleryou can show black is white by argument said filby
# time traveller

# 困惑度 1.1, 40239.9 词元/秒 cuda:0
# time traveller for so it will be convenient to speak of himwas e
# traveller wht said the psychologistnor having only length b

