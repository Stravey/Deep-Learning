import math
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from d2l import torch as d2l

# 初始化两个全为 1 的 10000维向量
n = 10000
a = torch.ones([n])
b = torch.ones([n])

# 定义计时器
class Timer:
    """"记录多次运行时间"""
    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    """启动计时器"""
    def start(self):
        self.tik = time.time()

    """停止计时器并将时间记录在列表中"""
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    """返回平均时间"""
    def avg(self):
        return sum(self.times) / len(self.times)

    """返回时间总和"""
    def sum(self):
        return sum(self.times)

    """返回累计时间"""
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

def method_a():
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')

def method_b():
    timer = Timer()
    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')

# 正态分布与平方损失
# 定义正态分布
def normal(x,mu,sigma):
    p = 1 / math.sqrt(2 *  math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)

def normal_print():
    x = np.arange(-7,7,0.01)
    # 三组不同的均值 方差
    params = [(0,1),(0,2),(3,1)]

    d2l.plot(x,[normal(x,mu,sigma) for mu,sigma in params],xlabel='x',
        ylabel='p(x)',figsize=(4.5,2.5),
        legend=[f'mean {mu},std{sigma}'for mu,sigma in params])

    plt.savefig('normal_distributional.png',dpi = 300,bbox_inches = 'tight')
    plt.show()

    
if  __name__ == '__main__':
    # method_a()
    # method_b()
    normal_print()
