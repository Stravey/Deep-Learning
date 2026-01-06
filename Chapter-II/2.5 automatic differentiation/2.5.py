# 自动微分是pytorch的一种自动计算导数和梯度的方法 无论函数多么复杂 都可以自动计算 更容易实现梯度下降算法
import torch
x = torch.arange(4.0,requires_grad=True)
# 默认为None
print(x.grad)

# 计算点积
y = 2 * torch.dot(x,x)
print(y)

# 计算梯度
y.backward()
print(x.grad)
print(x.grad == 4 * x)

# 默认情况下 pytorch会累计梯度 我们需要清楚之前的值
x.grad.zero_()
y = x.sum()
print(y)
y.backward()
print(x.grad)

# 非标量变量的反向传播
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# 分离计算
def func() :
    x = torch.arange(4.0,requires_grad=True)
    y = x * x
    u = y.detach()
    z = u * x
    z.sum().backward()
    print(x.grad == u)
    x.grad.zero_()
    y.sum().backward()
    print(x.grad == 2 * x)

def f(x):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c



if __name__ == '__main__':
    func()



