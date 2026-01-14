# 感知机 二分类模型 人工智能最早的模型 感知机引入了隐藏层 softmax只有输入层 输出层
# 深度神经网络的开端 感知机不能拟合XOR函数 求解方法等价于使用批量大小为1的梯度下降 导致AI的第一个寒冬
# 多层感知机 当前AI的主要模型 使用隐藏层和激活函数来得到非线性没模型
# 多类回归 就是softmax加 n 个隐藏层
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from tornado.concurrent import run_on_executor

# 激活函数
# 1.ReLU函数
x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(),'x','relu(x)',figsize=(5,2.5))
plt.show()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of relu',figsize=(5,2.5))
plt.show()

# 2.sigmoid函数
y = torch.sigmoid(x)
d2l.plot(x.detach(),y.detach(),'x','sigmoid(x)',figsize=(5,2.5))
plt.show()
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of sigmoid',figsize=(5,2.5))
plt.show()

# 3.tanh函数
y = torch.tanh(x)
d2l.plot(x.detach(),y.detach(),'x','tanh(x)',figsize=(5,2.5))
plt.show()
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of tanh',figsize=(5,2.5))
plt.show()
