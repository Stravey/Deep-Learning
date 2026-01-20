# 深度循环神经网络
import torch
from torch import nn
from d2l import torch as d2l

from RNN.RNNM import RNNModelScratch
from RNN.TP import load_data_time_machine

batch_size, num_steps = 32,35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers)
model = RNNModelScratch(lstm_layer,len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)

# 训练结果
# 困惑度 1.0, 112694.0 词元/秒 cuda:0
# time traveller for so it will be convenient to speak of himwas e
# travellerit to mentif ther tave asler three dimensions and
