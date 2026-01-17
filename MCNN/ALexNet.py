import torch
from torch import nn
from d2l import torch as d2l

import train

# 1.定义模型 构造器
net = nn.Sequential(
    # 使用一个11*11的更大窗口来捕捉对象
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口 使用填充为2来使得输入与输出的高和宽一致
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 相比LeNet增加三个卷积层
    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 展平
    nn.Flatten(),
    # 全连接层
    nn.Linear(6400,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    # 输出层
    nn.Linear(4096,10))

# X = torch.randn(1,1,224,224)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape: \t',X.shape)

batch_size = 128
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize = 224)

# 训练模型
lr,num_epochs = 0.01,10
train.train_model(net, train_iter, test_iter,num_epochs, lr, d2l.try_gpu())

