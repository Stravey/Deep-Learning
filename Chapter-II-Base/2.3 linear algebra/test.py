import torch

def test_01():
    print("test demo 01")
    A = torch.arange(9).reshape(3,3)
    B = A.clone()
    print(A)
    print(B.reshape(3,3))
    print(A == B)

def test_02():
    print("test demo 02")
    A = torch.arange(4).reshape(2,2)
    B = torch.randn(2,2)
    print(A)
    print(B)
    sum_AB = A.sum() + B.sum()
    summ = torch.add(A, B).sum()
    print(sum_AB == summ)
    print("\n")

# 如果矩阵 a 是对称的，则矩阵 a^T + a 也一定是对称的
# 但如果矩阵 a 不对称，那么矩阵 a^T + a 就一般不是对称的
def test_03():
    print("test demo 03")
    x = torch.randn(1,5)
    y = x.view(5,1)
    print(f"the sum value is\n {torch.add(x,y).numpy()}")
    print("\n")

def test_04():
    print("test demo 04")
    A = torch.randn(2,3,4)
    print(A)
    print(len(A))
    print("\n")

# len(A) 与特定轴长度相等 这个轴是axis = 0
def test_05():
    print("test demo 05")
    A = torch.randn(2,4)
    print(A.numpy())
    print(len(A))
    print("\n")

def test_06():
    print("test demo 06")
    A = torch.arange(4).reshape(2,2)
    print(A)
    print(A.sum(axis = 1))
    print(A / A.sum(axis = 1))
    print("\n")

# 则输出结果将是一个标量，即形状为 ()。
# 在 PyTorch 中，可以使用 dim 参数指定要沿着哪些轴进行求和，如果对所有的轴进行求和，
# 则可以简单地写为 tensor.sum()，输出结果的形状为 ()。
# 如果只沿着某个轴进行求和，则输出结果形状中会减少对应的维度。
# 例如，对轴 0 进行求和，则输出形状为 (3, 4)；对轴 1 进行求和，则输出形状为 (2, 4)
# 对轴 2 进行求和，则输出形状为 (2, 3)。
def test_07():
    print("test demo 07")
    A = torch.randn(2,3,4)
    print(A)
    output = A.sum(dim=(0,1,2))
    print(output)
    print("\n")

# linalg.norm 函数可以用于计算一个张量的范数，无论它具有多少个轴
# 当我们指定了轴时，函数会返回沿着指定轴计算得到的范数，输出形状会从张量中移除对应的轴
# 如果没有指定轴，则函数对整个张量进行计算，输出结果是一个标量
def test_08():
    print("test demo 08")
    A = torch.arange(24).reshape(2,3,4)
    print(A)
    print(torch.linalg.norm(A.float()))
    print("\n")


if __name__ == '__main__' :
    test_08()
    test_07()
    test_06()
    test_05()
    test_04()
    test_03()
    test_02()
    test_01()

