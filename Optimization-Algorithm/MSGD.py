# Minibatch Stochastic Gradient Descent
# Minibatch随机梯度下降

import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

from RNN.SM import features

# 1 Vectorization and Cashes
# Define three matrics
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)

class Timer: #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cum_sum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

# Create an instance
timer = Timer()

# Compute A = BC one element at a time  slow
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i,: ], C[:, j])
timer.stop()
# print(timer.stop())

# Compute A = BC one column at a time  fast
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
# print(timer.stop())

timer.start()
A = torch.mm(B, C)
timer.stop()
# 十亿次浮点运算每秒 gigaflops
gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f},'
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')

# 2 Minibatches
timer.start()
for j in range(0, 256, 64):
    A[:, j : j + 64] = torch.mm(B, C[:, j : j + 64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[2]:.3f}')


# 3 Reading the Dataset
#@save
# A dataset developed by NASA to test the wing noise from different aircraft
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')
#@save
# Only use 1500 examples
def get_data(batch_size = 10, n = 1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype = np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1] - 1

# 4 Implementation from Scratch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data_zero_()

#@save
def train(train_iter, states, hyperparams, data_iter,
          feature_dim, num_epochs = 2):

    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X : d2l.linreg(X, w, b), d2l.squared_loss

    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            # todo trainer_fn
            # trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss), ))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum() / num_epochs:.3f} sec / epoch')
    return timer.cumsum(), animator.Y[0]

# Train SGD
def train_sgd(lr, batch_size, num_epoches = 2):
    data_iter, feature_dim = get_data(batch_size)
    return train(sgd, None, {'lr': lr}, data_iter, feature_dim, num_epoches)

