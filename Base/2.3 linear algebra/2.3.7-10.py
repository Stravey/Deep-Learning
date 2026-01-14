import torch

# 计算点积使用dot()函数
x = torch.arange(4,dtype = torch.float32)
y = torch.ones(4,dtype= torch.float32)
print("x =", x)
print("y =", y)
# 法1
print("torch.dot(x, y) =", torch.dot(x, y))
# 法2
print("torch.dot(x, y) =", torch.dot(x.float(), y))

# 也可以使用求和计算点积
print("用sum计算点积:", torch.sum(x * y))

# pytorch中默认为float64
A = torch.arange(20,dtype=float).reshape(5,4)
print(A)
print(A.shape)
print(x.shape)
# 我们可以使用张量表示矩阵-向量积 使用mv函数
# 需要强制类型转换
print(torch.mv(A.float(),x))

# 矩阵乘法 5x4 * 4x3 = 5x3矩阵
B = torch.ones(4,3)
print(torch.mm(A.float(),B))

# 范数
# 深度学习中 L2 范数 就是 欧几里得距离 是向量元素平方和的平方根
# 深度学习中 L1 范数 就是 向量元素的绝对值之和
# L2 范数
u = torch.tensor([3.0,-4,0])
print(torch.norm(u))

# L1 范数
print(torch.abs(u).sum())

# 弗罗贝尼乌斯范数 是 矩阵元素的平方和 的 平方根 相当于矩阵形式的L2范数
print(torch.norm(torch.ones(4,9)))
