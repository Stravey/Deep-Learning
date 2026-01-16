# 多输入多输出通道

import torch
from d2l import torch as d2l

# 1.多输入通道
def corr2d_multi_in(X,K):
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

X = torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],
                  [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])
Y = torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0,2.0],[3.0,4.0]]])
print(corr2d_multi_in(X,Y))

# 2.多输出通道
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
# 通过将核张量K与K+1（K中每个元素加）和K+2连接起来，构造了一个具有个输出通道的卷积核
K = torch.stack((K,K + 1,K + 2),0)
print(K.shape)
print(corr2d_multi_in_out(X,K))

# 3.1x1卷积核
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i,h * w))
    K = K.reshape((c_o,c_i))
    # 全连接层矩阵乘法
    Y = torch.matmul(K,X)
    return Y.reshape((c_o,h,w))
X = torch.normal(0,1,(3,3,3))
Y = torch.normal(0,1,(2,3,1,1))
Y1 = corr2d_multi_in_out_1x1(X,K)
Y2 = corr2d_multi_in_out(X,K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
