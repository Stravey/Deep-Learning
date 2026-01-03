import torch
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
# **求幂运算
print(x ** y)

print(torch.exp(x))
print(torch.log(x))

# 将多个张量连接在一起 我们只需提供张量列表 并给出沿哪个轴连接
X = torch.arange(12,dtype = torch.float32).reshape((3,4))
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
# 按行 轴0
New_X = torch.cat((X,Y),dim = 0)
# 按列 轴1
New_Y = torch.cat((X,Y),dim = 1)
print("连接前:")
print(X)
print(Y)

print("连接后:")
print(New_X)
print(New_Y)

print(X == Y)

print("总和:")
total = X.sum()
print(total)