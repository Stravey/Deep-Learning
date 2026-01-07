# 节省内存
import torch
X = torch.arange(4).reshape((2,2))
print(X)
Y = torch.arange(2).reshape((1,2))
print(Y)

# 我们用Python的id()函数 id()函数提供了内存中引用对象的确切位置
before = id(Y)
# 运行Y = Y + X 我们会发现id(Y)指向另一个位置
# 这是因为Python首先计算Y + X 为结果分配新的内存 然后使Y指向内存中的这个新位置
Y = Y + X
print(id(Y) == before)


Z = torch.zeros_like(Y)
print("id(Z) = ", id(Z))
# 我们可以使用切片表示法将操作的结果分配给先前分配的数组
Z[:] = X + Y
print("id(Z) = ", id(Z))

before = id(X)
X += Y
print(id(X) == before)
