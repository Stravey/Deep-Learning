# 转换为其他Python对象
import torch
x = torch.arange(6).reshape(2,3)
print(x)
# 将深度学习框架定义的张量转换为NumPy张量（ndarray）
A = x.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

# 要将大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))

# 深度学习存储和操作数据的主要接口是张量,其实就是n维数组
# 它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他Python对象。