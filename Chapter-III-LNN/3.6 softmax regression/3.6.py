# 手搓softmax回归
# 多分类处理算法

import torch
from IPython import display
from d2l import torch as d2l

# 每次随机选取256图片
batch_size = 256
# 返回训练集 测试集的迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1.初始化模型参数
num_inputs = 784
# 网络输出维度为10
num_outputs = 10
# 初始化权重
W = torch.normal(0,0.01,size = (num_inputs,num_outputs),requires_grad=True)
# 偏移
b = torch.zeros(num_outputs,requires_grad=True)

# 2.定义softmax操作
# X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# print(X.sum(0,keepdim=True),"\n",X.sum(1,keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    # 使用广播机制 对对应元素进行相除
    return X_exp / partition
X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
# print(X_prob)
# print(X_prob.sum(1))

# 3.定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W) + b)

# 4.定义损失函数 多分类的交叉熵损失函数
y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# print(y_hat[[0,1],y])
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)),y])
cross_entropy(y_hat, y)

# 5.分类精度 预测值y_hat 真实值y
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
# print(accuracy(y_hat, y) / len(y))   0.5

# 定义一个实用程序类Accumulator
class Accumulator:
    def __init__(self,n):
        self.data = [0.0] * n

    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 对于任意数据迭代器data_iter可访问的数据集，我们可以评估在任意模型net的精度
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]

# 6.训练
def train_epoch_ch3(net,train_iter,loss,updater):
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

class Animator:
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

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
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

# 定义优化器
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)

# 开始训练
num_epochs = 10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)

# 7.预测
def predict_ch3(net,test_iter,n = 6):
    global X, y
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true,pred in zip(trues,preds)]
    d2l.show_images(
        X[0:n].reshape((n,28,28)),1,n,titles = titles[0:n])

predict_ch3(net,test_iter)

# 用时1min 数据量54MB
