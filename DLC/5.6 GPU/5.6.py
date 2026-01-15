import torch
import sys
from torch import nn

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 默认张量存储在cpu上
x = torch.tensor([1,2,3])
print(x.device)
