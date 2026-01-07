# 实现一个梯度下降算法 计算 x^2 + y^2 的极小值

import torch

def f(x,y):
    return x ** 2 + y ** 2

if __name__ == '__main__':
    x = torch.tensor([1.1], requires_grad=True)
    y = torch.tensor([2.1], requires_grad=True)
    n = 100
    # 迭代速率 每一步移动距离
    alpha = 0.05

    for i in range(1, n + 1):
        z = f(x, y)
        z.backward()

        x.data -= alpha * x.grad.data
        y.data -= alpha * y.grad.data

        x.grad.zero_()
        y.grad.zero_()

        print(f'After {i} step,'
              f'x = {x.item():.3f},'
              f'y = {y.item():.3f},'
              f'z = {z.item():.3f}')
