import matplotlib.pyplot as plt
import torch
from matplotlib.lines import lineStyles
from mpmath.functions.zetazeros import count_to
from sympy.logic.algorithms.dpll import dpll_int_repr
from torch.distributions import multinomial
from d2l import torch as d2l

# 简单来说 机器学习就是做出预测 概率是构建深度学习模型的基础
# 例子 : 根据照片区分猫和狗  首要问题就是分辨率不同所引起的难题
# 大数定律

# 模拟掷骰子
# 传入概率向量  输出是另一个相同长度的向量 它在索引i处的值是采样结果中i出现的次数
fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1,fair_probs).sample())
print("---------------")
# 生成多个样本
print(multinomial.Multinomial(10,fair_probs).sample())
print("---------------")
# 模拟1000次投掷
counts = multinomial.Multinomial(1000,fair_probs).sample()
# 相对频率作为估计值
print(counts / 1000)

# 模拟500组实验
counts = multinomial.Multinomial(10,fair_probs).sample((500,))
cum_counts = counts.cumsum(dim = 0)
estimates = cum_counts / cum_counts.sum(dim = 1,keepdim = True)

d2l.set_figsize((6,4.5))
for i in range(6):
    d2l.plt.plot(estimates[:,i].numpy(),
                 label=("P(die = " + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167,color='black',linestyle='dashed')
d2l.plt.gca().set_xlabel("Groups of experiments")
d2l.plt.gca().set_ylabel("Estimated probability")
d2l.plt.legend()
# 将图片保存到当前目录下
plt.savefig("dice_simulation.png",dpi = 300,bbox_inches='tight')

d2l.plt.show()
