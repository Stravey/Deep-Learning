# 训练误差 模型在训练数据上的误差
# 泛化误差 模型在新数据上的误差
# 验证数据集  测试数据集（只能用一次）
# K-则交叉验证
# 欠拟合 过拟合
# 模型容量 拟合各种函数的能力
# 给定模型种类 两个主要因素：参数个数 参数值的选择范围
# VC维 统计学习理论的一个核心思想 支持 n 维输入的感知机的VC维是 n + 1
# 一些多层感知机的VC维为nlog2n
# 数据复杂度 样本个数 每个样本的元素个数 时间、空间结构 多样性

# 多项式回归
import torch
import numpy as np
import math
from IPython import display
from d2l.torch import Accumulator
from torch import nn
from d2l import torch as d2l

# 生成数据集
max_degree = 20
n_train,n_test = 100,100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size = (n_train + n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i + 1)
labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale = 0.1,size = labels.shape)

true_w,features,poly_features,labels = [torch.tensor(x,dtype=torch.float32)
                                        for x in [true_w,features,poly_features,labels]]
# print(features[:2],'\n',poly_features[:2,:],'\n',labels[:2])


def accuracy(y_hat, y):  #@save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 对于任意数据迭代器data_iter可访问的数据集，我们可以评估在任意模型net的精度
def evaluate_accuracy(net,data_iter):   #@save
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net,train_iter,loss,updater):   #@save
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()
    # 训练损失总和 训练准确度综合 样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            # pytorch内部优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 定制优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum().detach()),accuracy(y_hat,y),y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2],metric[1] / metric[2]

class Animator:   #@save
    """在动画中绘制数据"""
    def __init__(self,xlabel = None,ylabel = None,legend = None,xlim = None,
                 ylim = None,xscale = 'linear',yscale = 'linear',
                 fmts = ('-','m--','g-','r:'),nrows = 1,ncols = 1,
                 figsize = (3.5,2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig,self.axes = d2l.plt.subplots(nrows,ncols,figsize = figsize)
        if(nrows * ncols == 1):
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        self.X,self.Y,self.fmts = None,None,fmts

    def add(self,x,y):
        # 向图表中添加多个数据点
        if not hasattr(y,"__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x,"__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_epoch(net,train_iter,test_iter,loss,num_epochs,updater):   #@save
    global test_acc, train_metrics
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],
                        legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch + 1,train_metrics + (test_acc,))
    train_loss,train_acc = train_metrics
    assert train_loss < 0.5,train_loss
    assert train_acc <= 1 and train_acc > 0.7,train_acc
    assert test_acc < 1 and test_acc > 0.7,test_acc

# 训练测试
def evaluate_loss(net,data_iter,loss):
   metric = d2l.Accumulator(2)
   for X,y in data_iter:
       out = net(X)
       y = y.reshape(out.shape)
       l = loss(out,y)
       metric.add(l.sum(),l.numel())
   return metric[0] / metric[1]

def train(train_features,test_features,train_labels,test_labels,
          num_epochs = 400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train=False)
    trainer  = torch.optim.SGD(net.parameters(),lr=0.01)
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',
                            xlim=[1,num_epochs],ylim=[1e-3,1e2],
                            legend=['train','test'])
    for epoch in range(num_epochs):
        # 训练一个epoch
        if isinstance(net, torch.nn.Module):
            net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()

        # 每20个epoch记录一次损失
        if epoch == 0 or (epoch + 1) % 20 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))

    print('weight:', net[0].weight.data.numpy())
    return net

print("=== 使用前4个多项式特征 ===")
train(poly_features[:n_train,:4],poly_features[n_train:,:4],
      labels[:n_train],labels[n_train:])

print("\n=== 使用前2个多项式特征 ===")
train(poly_features[:n_train,:2],poly_features[n_train:,:2],
      labels[:n_train],labels[n_train:])

print("\n=== 使用所有多项式特征 ===")
train(poly_features[:n_train,:],poly_features[n_train:,:],
      labels[:n_train],labels[n_train:],num_epochs=1500)


