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