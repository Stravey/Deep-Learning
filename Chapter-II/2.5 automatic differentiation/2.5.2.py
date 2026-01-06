import torch
# 向量输入的正确处理方法
print("向量输入的正确方法")

print("\n方法1:使用sum()")
x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
y = x ** 2
y_sum = y.sum()
y_sum.backward()
print(x)
print(y)
print(y_sum.item())
print(x.grad)

# 重置梯度
x.grad.zero_()
print("\n方法2.传递梯度参数")
y_sum2 = x ** 2
grad_output = torch.tensor([1.0,1.0,1.0])
y_sum2.backward(gradient=grad_output)
print(x)
print(y_sum2)
print(grad_output)
print(x.grad)

