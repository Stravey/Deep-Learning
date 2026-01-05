import torch
from sympy import true

# 降维
x = torch.arange(4,dtype = torch.float32)
print(x)
# 求对应张量的元素和 向量
print(x.sum())

# 求对应张量的元素和 矩阵
A = torch.arange(20,dtype = torch.float32).reshape(5,4)
print(A)
print(A.shape)
print(A.sum())

# 默认情况下 调用求和函数会沿所有的轴降低张量的维度 使其变成一个标量
# 我们还可以通过指定张量沿哪一个轴通过求和降低维度
# 轴 0 按行分别求和 变成一维行向量
A_sum_axis0 = A.sum(axis = 0)
print(A_sum_axis0)
print(A_sum_axis0.shape)

# 轴 1 按列分别求和 变成一维列向量
A_sum_axis1 = A.sum(axis = 1).reshape(5,1)
print(A_sum_axis1)

# 沿着行和列对矩阵求和 等价于对矩阵的所有元素求和
print(A.sum(axis = [0,1]))

# 求平均值函数mean() 或使用sum() / numel()
print(A.mean())
print(A.sum() / A.numel())

# 求平均值函数也可以沿指定轴降低张量的维度
print(A.mean(axis = 0))
print(A.sum(axis = 0) / A.shape[0])

sum_A = A.sum(axis = 1,keepdim=True)
print(sum_A)

print(A / sum_A)

# 按行
print(A.cumsum(axis = 0))