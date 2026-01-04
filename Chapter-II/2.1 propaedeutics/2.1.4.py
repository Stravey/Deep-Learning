# 索引和切片
import torch
from numpy.ma.core import indices

# 1.一维张量
x = torch.arange(12)
print("一维初始张量:")
print(x)

# 一维索引
print("\n一维张量索引:")
print(x[3])
# -1 最后一个元素
print(x[-1])

# 一维切片
print("\n一维张量切片:")
# 左闭右开
print(x[2:5])
# 从头开始
print(x[:5])
# 到末尾
print(x[5:])
# 步长为2
print(x[::2])

# 2.二维张量  矩阵
x = torch.arange(12).reshape(3, 4)
print("\n初始二维张量:")
print(x)

# 二维索引
print("\n二位张量索引:")
# 第二行
print(x[1])
# 第2行第3列元素
print(x[1,2])
# 最后一行
print(x[-1])

# 二维切片
print("\n二维张量切片:")
# 第2行到第3行
print(x[1:3])
# 所有行的第2列到第3列
print(x[:,1:3])
# 行步长为2 列步长为2
print(x[::2,::2])

# 3.高级切片操作
print("\n高级切片操作:")
# 列表索引
a = torch.tensor([0,2])
print(x[a])

# 布尔索引 布尔掩码
mask = x > 5
print(mask)
print(x[mask])