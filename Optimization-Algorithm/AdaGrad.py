import torch
import math
from d2l import torch as d2l
from matplotlib import pyplot as plt

# AdaGrad instance
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2 # 计算梯度
    s1 += g1 ** 2  # 累计梯度平方
    s2 += g2 ** 2  # 累计梯度平方
    x1 -= eta / math.sqrt(s1 + eps) * g1 # 自适应更新 会加一个误差值eps
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

# x1 缓慢  x2 陡峭
def f_2d(x1, x2):
    """objective function"""
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
# plt.show()
