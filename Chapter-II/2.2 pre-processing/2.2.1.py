# 要想用深度学习来解决现实问题 要先从处理原始数据开始 而不是已准备好的张量格式数据
# 处理数据一般使用pandas包 可以和张量很好的兼容
import os
import pandas as pd

# 在当前目录创建data文件夹
os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')

# 创建csv文件
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取数据
data = pd.read_csv(data_file)
print(data)

