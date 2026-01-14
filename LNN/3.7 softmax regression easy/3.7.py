# 使用pytorch中的nn实现softmax

import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1.初始化模型参数
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std = 0.01)
net.apply(init_weights)

# 2.softmax实现
loss = nn.CrossEntropyLoss(reduction='none')

# 3.优化算法
# 学习率为0.1的小批量随机梯度下降
trainer = torch.optim.SGD(net.parameters(),lr = 0.01)

# 4.训练
num_epochs = 10
d2l.train_ch3(net,train_iter,test_iter,num_epochs,trainer)
