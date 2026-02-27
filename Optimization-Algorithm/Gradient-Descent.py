# Gradient Descent
import numpy as np
import torch
from d2l import torch as d2l
from sympy.physics.control.control_plots import plt
from torchgen.dest.native_functions import torch_api_key_word_prefix


# 1 One-Dimensional Gradient Descent
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x

def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])
show_trace(results, f)

# Learning Rate
show_trace(gd(0.05, f_grad), f)
show_trace(gd(1.1, f_grad), f)


# Local Minima
c = torch.tensor(0.15 * np.pi)
# Objective function
def f(x):
    return x * torch.cos(c * x)

# Gradient of the objective function
def f_grad(x):
    return torch.cos(c * x) - c * x * torch.sin(c * x);
show_trace(gd(2, f_grad), f)

# 2 可视化多元梯度下降优化
# Multivariate Gradient Descent
def train_2d(trainer, steps = 20, f_grad = None):
    """Optimize a 2D objective function with a customized trainer."""
    # 's1' and 's2' are internal state variables that will be used in Momentum, Adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

def f_2d(x1, x2): # Objective function
    return x1 ** 2 + 2 * x2 * 2

def f_2d_grad(x1, x2): # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))

# 3 Adaptive Methods
# 3.1 Newton's Method
# 3.1.1 Example 1
c = torch.tensor(0.5)
def f(x): # Objective function
    return torch.cosh(c * x)

def f_grad(x): # Gradient of the objective function
    return c * torch.sinh(c * x)

def f_hess(x): # Hessian of the objective function
    return c ** 2 * torch.cosh(c * x)

def newton(eta = 1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results
show_trace(newton(), f)

# 3.1.2 Example 2
c = torch.tensor(0.15 * np.pi)

def f(x): # Objective function
    return x * torch.cos(c * x)

def f_grad(x): # Gradient of the objective function
    return torch.cos(c * x) - c * x * torch.sin(c * x)

def f_hess(x): # Hessian of the objective function
    return - 2 * c * torch.sin(c * x) - x * c ** 2 * torch.cos(c * x)

show_trace(newton(), f)

show_trace(newton(0.5), f)
