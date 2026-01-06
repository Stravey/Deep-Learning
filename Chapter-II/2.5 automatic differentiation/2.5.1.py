# 求导是几乎所有深度学习优化算法的关键步骤
# 深度学习框架通过自动计算导数 即自动微分 加快求导
# 自动微分使系统能够随后反向传播梯度 反向传播意味着跟踪整个计算图 填充关于每个参数的偏导数
import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.tensor(3.0, requires_grad=True)  # 创建一个需要梯度的标量张量
y = 2 * x ** 2 + 3 * x + 1  # 定义一个二次函数

# 计算y的梯度
y.backward()

print(f"函数: y = 2x² + 3x + 1")
print(f"在 x = {x.item()} 处:")
print(f"  y = {y.item():.2f}")
print(f"  dy/dx = {x.grad.item():.2f}")
print(f"  理论值: 4*{x.item()} + 3 = {4*x.item()+3:.2f}")

def f(x):
    return 2 * x ** 2 + 3 * x + 1

def derivative_f(x):
    return 4 * x + 3

# 创建数据
x_vals = np.linspace(-3,3, 100)
y_vals = f(x_vals)
y_derivative = derivative_f(x_vals)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 子图1: 函数和切线
x0 = 3.0
y0 = f(x0)
slope = derivative_f(x0)
tangent = slope * (x_vals - x0) + y0

axes[0].plot(x_vals, y_vals, 'b-', label='y = 2x² + 3x + 1', linewidth=2)
axes[0].plot(x_vals, tangent, 'r--', label=f'切线 (x={x0})', linewidth=1.5)
axes[0].scatter([x0], [y0], color='red', s=50, zorder=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('函数和切线')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.2)
axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.2)

# 子图2: 导数
axes[1].plot(x_vals, y_derivative, 'g-', label="y' = 4x + 3", linewidth=2)
axes[1].scatter([x0], [slope], color='red', s=50, zorder=5)
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.2)
axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.2)
axes[1].set_xlabel('x')
axes[1].set_ylabel("y'")
axes[1].set_title('导数函数')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

