# 图像分类 入门
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 将图像设置为svg
d2l.use_svg_display()

# MNIST数据集是图像分类中广泛使用的数据集之一,但作为基准数据过于简单
# 于是1使用类似但更复杂的Fashion-MNIST数据集
# 将图像转换为张量格式
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root = "../data",train=True,transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root = "../data",train=False,transform=trans,download=True)
# 训练集样本数：60000
a = len(mnist_train)
# 训练集样本数：60000
b = len(mnist_test)
# 单个样本的形状：[1, 28, 28]
c = mnist_train[0][0].shape

# Fashion-MNIST中包含10个类别
# 标签映射函数
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat',
                   'sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

# 可视化样本
def show_images(imgs,num_rows,num_cols,titles = None,scale = 1.5):
    figsize = (num_cols * scale,num_rows * scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
# 显示图像
plt.show()

batch_size = 256
def get_dataloader_workers():
    return 4
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

