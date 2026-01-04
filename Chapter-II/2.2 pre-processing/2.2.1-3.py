# 要想用深度学习来解决现实问题 要先从处理原始数据开始 而不是已准备好的张量格式数据
# 处理数据一般使用pandas包 可以和张量很好的兼容
import os
import pandas as pd
import torch

# 1.读取数据集
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
print("原始数据:")
print(data)
print("\n")


# 2.处理缺失值 插值法和删除法
# 分离输入输出
# 位置索引iloc 我们可以将data分为inputs和outputs inputs为data前两列 outputs为data最后一列
# 对于inputs中的缺失值 我们用同一列的均值替换NaN项
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]

# 处理数值列的缺失值（只对NumRooms列计算平均值）
# 方法1：指定列名
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print("处理后的inputs:")
# 方法2：或者只选择数值列
# numeric_inputs = inputs.select_dtypes(include=['number'])
# inputs[numeric_inputs.columns] = numeric_inputs.fillna(numeric_inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs,dummy_na=True)
print(inputs)
print("数据类型:",inputs.dtypes)

# x = torch.tensor(inputs.values)
# y = torch.tensor(outputs.values)
# 不能使用上述进行张量输出 因为张量亦被转换为布尔类型 而要输出需转换为数值类型
x = torch.tensor(inputs.values.astype(float))
y = torch.tensor(outputs.values.astype(float))
print(x)
print(y)
