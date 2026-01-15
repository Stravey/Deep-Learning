# MLP处理图像分类时会有问题 参数值太大 大于现存值 故需要解决此问题
# 设计计算机视觉的神经网络遵循的原则
# 1.平移不变性   2.局部性

# 图像卷积 二维
import torch
from torch import nn

# 计算二维互相关运算
def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i : i + h,j : j + w] * K).sum()
    return Y
X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
K = torch.tensor([[0.0,1.0],[2.0,3.0]])
# print(corr2d(X,K))

# 卷积层
class Conv2D(nn.Module):
    def __init__(self,kernal_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernal_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones(6,8)
X[:,2 : 6] = 0
print(X)
print("-----------------")
K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
print(Y)
print("-----------------")
print(corr2d(X.t(),K))

# 学习卷积核
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)

X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1,2)))
