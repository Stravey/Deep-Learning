import torch
from torch import nn
from d2l import torch as d2l

from MSGD import get_data

def train_concise(trainer_fn, hyperparams, data_iter, num_epoches=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epoches], ylim=[0.22, 0.35])

    n, timer = 0, d2l.Timer()
    for _ in range(num_epoches):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                """"""
                animator.add(n / X.shape[0] / len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum() / num_epoches:.3f} sec / epoch')

data_iter, _ = get_data(10)
trainer = torch.optim.SGD
train_concise(trainer, {'lr': 0.01}, data_iter)
