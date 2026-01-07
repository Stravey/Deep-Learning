import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,NullFormatter
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width

# question 1
# 计算二阶导数要在一阶导数的基础上 故开销更大

# question 2
x = torch.arange(40.,requires_grad=True)
y = 2 * torch.dot(x ** 2,torch.ones_like(x))
y.backward()
print(x.grad)

# question 3
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# question 4
def func(a):
    b = a ** 2 + abs(a)
    c = b ** 3 - b * (-3)
    return c

def test_func():
    a = torch.randn(size=(3,1),requires_grad=True)
    print(a.shape)
    print(a)
    d = f(a)
    d.sum().backward()
    print(a.grad)

# question 5
# 计算sin(x)和它的梯度
def paint():
    f, ax = plt.subplots(1,figsize=(10,6))

    x = np.linspace(-np.pi,np.pi,100)
    x1 = torch.tensor(x,requires_grad=True)

    # 计算sin(x)和它的梯度
    y1 = torch.sin(x1)
    y1.sum().backward()

    # 绘制sin(x)和它的导数cos(x)
    ax.plot(x,np.sin(x),label="sin(x)")
    ax.plot(x,x1.grad,label="gradient of sin(x) = cos(x)",linewidth=2)

    # 设置横坐标以π为单位
    ax.set_xticks([-np.pi,-np.pi / 2,0,np.pi / 2,np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    # 设置网格和标签
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (in radians)', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('sin(x) and its derivative cos(x)', fontsize=14)

    # 添加图例
    ax.legend(loc="upper left",shadow=True,fontsize=12)

    # 添加零线参考
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    x = torch.randn(size=(3,1),requires_grad=True)
    d = f(x)
    # d.backward() ----> runtime error d不是标量 需要先求和在计算梯度
    d.sum().backward()
    print(d)
    print("-------\n")
    test_func()
    paint()
