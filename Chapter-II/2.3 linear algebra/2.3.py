import torch

# 1.定义标量 0维张量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x.dtype)
print(y.dtype)
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)

# 2.定义向量 一维张量
x = torch.arange(4)
print(x)
print(x[3])
# 可用python内置函数len访问张量的长度
print("向量长度:",len(x))
# 当用张量表示一个向量时 我们也可以通过 shape属性访问向量的长度
print(x.shape)

# 3.定义矩阵 矩阵将向量从一维推广到二维
A = torch.arange(20).reshape(5,4)
print("矩阵A:",A)
# 输出矩阵的转置
print("A转置:",A.T)
# 对称矩阵
B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B)
print("矩阵B:",B)
print("B转置:",B.T)
print(B == B.T)

# 4.张量 机器学习中非常重要的词
# 向量是一阶张量 矩阵是二阶张量
# 张量用特殊字体的大写字母表示 X Y Z  当进行图像处理时 张量变得非常重要
# 图像以n维数组形式出现 其中3个轴分别对应高度、宽度、以及一个通道用于表示颜色通道
X = torch.arange(24).reshape(2,3,4)
print(X)

# 5.张量算法的基本性质
# 元素从0到19 5 x 4矩阵
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
print(A)
# B拷贝一份
B = A.clone()
# 矩阵相加
print(A + B)
# 矩阵相乘
print(A * B)

a = 2
X = torch.arange(24).reshape(2,3,4)
print(a + X)
print((a + X).shape)