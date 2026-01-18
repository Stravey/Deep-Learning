# ResNet
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

import train

# 定义残差块
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,
                 use_1x1conv=False,strides=1):
        super().__init__()
        # 两个3x3卷积层
        self.conv1 = nn.Conv2d(input_channels,num_channels,
                               kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels,num_channels,
                               kernel_size=3,padding=1)
        # 短路连接 直接跳过中间几个卷积层 可选的1×1卷积
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,
                                   kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        # batch_norm层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self,X):
        # 主路径：Conv1 -> BN1 -> ReLU -> Conv2 -> BN2
        # 第一层卷积
        Y = F.relu(self.bn1(self.conv1(X)))
        # 第二层卷积
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 残差连接：主路径输出 + 调整后的输入
        Y += X
        return F.relu(Y)
# 验证结果
blk = Residual(3,3)
X = torch.randn(4,3,6,6)
Y = blk(X)
print(Y.shape)
blk = Residual(3,6,use_1x1conv=True,strides=2)
print(blk(X).shape)

# ResNet模型
b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                   nn.BatchNorm2d(64),nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels,num_channels,
                                use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))

net = nn.Sequential(b1,b2,b3,b4,b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(512,10))

X = torch.randn(size = (1,1,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

# 训练
lr,num_epochs,batch_size = 0.05,10,256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
train.train_model(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
