# 层 块
import torch
from torch import nn
import torch.nn.functional as F

net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.randn(2,20)
# print(net(X))

# 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256) # 隐藏层
        self.output = nn.Linear(256,10) # 输出层

    # 定义模型前向传播 如何根据输入X返回所需的模型输出
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

# 顺序块
# 定义两个函数
# 一个是将块逐个追加到列表中的函数
# 一个是前向传播函数 用于将输入按追加快的顺序传递给块组成的"链条”
class MySequential(nn.Sequential):
    def __init__(self,*args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self,X):
        for block in self._modules.values():
            X = block(X)
        return X
# 实现多层感知机
net1 = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

# 前向传播函数中执行代码 前向传播可以自定义
# 反向传播都是自动微分
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad = False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X,self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum > 1:
            X /= 2
        return X.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print(chimera(X))

# 一个块可以由许多层组成；一个块可以由许多块组成
# 块可以包含代码
# 块负责大量的内部处理，包括参数初始化和反向传播
# 层和块的顺序连接由Sequential块处理
