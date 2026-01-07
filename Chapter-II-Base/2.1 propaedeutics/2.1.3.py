import torch

#  广播机制
#  (1)通过适当复制元素扩展一个或两个数组 以便在转换之后 两个张量具有相同的形状
#  (2)对生成的数字执行按元素操作

# 相当于对行向量打转置
a = torch.arange(3).reshape(3,1)
print(a)

b = torch.arange(2).reshape(1,2)
print(b)

# a + b 相加时 若形状不统一 需要将其扩充 a - b 和 a * b类似
print(a+b)
print(a-b)
print(a*b)