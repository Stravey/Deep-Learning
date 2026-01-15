import torch
from torch import nn
from torch.nn import functional as F

# 1.加载和保存张量
x = torch.arange(4)
# 保存在当前文件夹
torch.save(x,'x-file')
# 将文件读回内存
x2 = torch.load('x-file')
print(x2)

# 存储张量
y = torch.zeros(4)
torch.save([x,y],'x-files')
x2,y2 = torch.load('x-files')
print(x2,y2)

# 读取或写入模型中的所有权重
mydict = {'x': x, 'y': y}
torch.save(mydict,'mydict')
mydict_x = torch.load('mydict')
print(mydict_x)


# 2.加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size = (2,20))
Y = net(X)

torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)
