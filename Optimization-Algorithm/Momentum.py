import torch
from matplotlib import pyplot as plt
from d2l import torch as d2l

from RNN.SM import features

# 动量Momentum可以帮助梯度下降在相关方向上加速收敛，同时在无关方向上抑制振荡
# 1 2D Visualization of Gradient Descent vs Momentum
# 第一部分：梯度下降和动量的2D可视化对比

eta = 0.4  # Learning rate / 学习率

# Define a 2D quadratic function with different curvatures in x1 and x2 directions
# 定义一个在x1和x2方向上有不同曲率的2D二次函数
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2  # Steep in x2 direction, gentle in x1 direction
                                          # x2方向陡峭，x1方向平缓

# Standard gradient descent (no momentum)
# 标准梯度下降（无动量）
def gd_2d(x1, x2, s1, s2):
    # s1, s2 are state variables (unused in standard GD)
    # s1, s2是状态变量（在标准GD中不使用）
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# Visualize gradient descent trajectory with learning rate 0.4
# 可视化学习率为0.4的梯度下降轨迹
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

eta = 0.6  # Increase learning rate / 增加学习率
# Visualize gradient descent trajectory with learning rate 0.6
# 可视化学习率为0.6的梯度下降轨迹
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

# Momentum-based gradient descent
# 基于动量的梯度下降
def momentum_2d(x1, x2, v1, v2):
    # Update velocity (momentum) terms
    # 更新速度（动量）项
    v1 = beta * v1 + 0.2 * x1  # Gradient in x1 direction (0.2 * x1)
    v2 = beta * v2 + 4 * x2     # Gradient in x2 direction (4 * x2)
    # Update parameters using velocity
    # 使用速度更新参数
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5  # Learning rate 0.6, momentum coefficient 0.5
                       # 学习率0.6，动量系数0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

eta, beta = 0.6, 0.25  # Learning rate 0.6, momentum coefficient 0.25
                        # 学习率0.6，动量系数0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

# 2 Visualizing Momentum Decay
# 第二部分：可视化动量衰减

d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]  # Different momentum coefficients / 不同的动量系数
for beta in betas:
    x = torch.arange(40).detach().numpy()
    d2l.plt.plot(x, beta ** x, label = f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend()
# plt.show()

# 3 Implementing Momentum for Linear Regression
# 第三部分：为线性回归实现动量优化

def init_momentum_states(feature_dim):
    """Initialize momentum states (velocity) for weights and bias"""
    """初始化权重和偏置的动量状态（速度）"""
    v_w = torch.zeros((feature_dim, 1))  # Velocity for weights / 权重的速度
    v_b = torch.zeros(1)                  # Velocity for bias / 偏置的速度
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    """Momentum SGD update rule"""
    """动量SGD更新规则"""
    for p, v in zip(params, states):
        with torch.no_grad():
            # Update velocity: v = momentum * v + gradient
            # 更新速度：v = 动量系数 * v + 梯度
            v[:] = hyperparams['momentum'] * v + p.grad
            # Update parameters: p = p - learning_rate * v
            # 更新参数：p = p - 学习率 * v
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()  # Clear gradients / 清空梯度

def train_momentum(lr, momentum, num_epochs=2):
    """Train using momentum SGD"""
    """使用动量SGD进行训练"""
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

# Load data / 加载数据
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# Experiment with different hyperparameters / 尝试不同的超参数
# lr = 0.02 momentum = 0.5
train_momentum(0.02, 0.5)

# lr = 0.01 momentum = 0.9
train_momentum(0.01, 0.9)

# lr = 0.005 momentum = 0.9
train_momentum(0.005, 0.9)

# 4 Using PyTorch's Built-in SGD with Momentum
# 第四部分：使用PyTorch内置的带动量的SGD

trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)

# 5 Visualizing Gradient Descent for Different Curvatures
# 第五部分：可视化不同曲率下的梯度下降

lambdas = [0.1, 1, 10, 19]  # Different curvature parameters / 不同的曲率参数
eta = 0.1  # Learning rate / 学习率
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = torch.arange(20).detach().numpy()
    d2l.plt.plot(t, (1 - eta * lam) ** t, label = f'lambda {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend()