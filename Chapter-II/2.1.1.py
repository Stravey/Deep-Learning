import torch
# 可以使用arange创建一个行向量
x = torch.arange(2,3)
# 可以用张量的shape属性访问张量形状
# x.shape
x.numel()
x = torch.arange(12)
# 可以用reshape来改变一个张量的形状而不改变数量和元素值
X = x.reshape(3,4)  # x.reshape(-1,4) x.reshape(3,-1) 与其等价
print(X)

# 创建一个形状为(2,3,4)的张量 元素全为0
y = torch.zeros(2,3,4)
print(y)

# 创建一个形状为(2,3,4)的张量 元素全为1
z = torch.ones(2,3,4)
print(z)


# 创建一个形状为(3,4)的张量 均服从0~1标准正态分布(高斯分布)
o = torch.randn(3,4)
print(o)
# 创建一个形状为(2,3)的张量 均服从0~1标准正态分布(高斯分布)
o = torch.randn(2,3)
print(o)