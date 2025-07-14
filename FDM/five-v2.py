import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def exact_sol(x, y):
    output = np.sin(np.pi * x) * np.sin(np.pi * y)
    return output


def rhs_f(x, y):
    output = -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    return output


def mesh_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    [X, Y] = np.meshgrid(x, y)
    return X, Y


def solve_poisson_equation(x0, x1, y0, y1, hx):
    #    n: 网格在 x 和 y 方向上的节点数（假设相等）

    # 初始化稀疏矩阵A和右侧向量b
    n = round(1 / h)
    A = lil_matrix((n ** 2, n ** 2))
    b = np.zeros(n ** 2)

    # 设置矩阵A中的系数和右侧向量b
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # 设置中心节点的系数
            index_inner = i * n + j
            A[index_inner, index_inner] = -4 / h ** 2

            # 设置相邻节点的系数
            A[index_inner, (i - 1) * n + j] = 1 / h ** 2  #
            A[index_inner, (i + 1) * n + j] = 1 / h ** 2  #
            A[index_inner, index_inner - 1] = 1 / h ** 2  #
            A[index_inner, index_inner + 1] = 1 / h ** 2  #

            # 设置右侧向量b中的值
            b[i * n + j] = rhs_f(i * h, j * h)  # 使用节点坐标来计算源项函数值

    # 将边界条件应用到矩阵A中
    for i in range(n):
        A[i * n, i * n] = 1  # 左
        b[i * n] = exact_sol(0, i * h)  # 设置边界条件对应的精确解值

        A[i * n + n - 1, i * n + n - 1] = 1  # 右
        b[i * n + n - 1] = exact_sol(1, i * h)

        A[i, i] = 1  # 下
        b[i] = exact_sol(i * h, 0)

        A[(n - 1) * n + i, (n - 1) * n + i] = 1  # 上
        b[(n - 1) * n + i] = exact_sol(i * h, 1)  # 设置边界条件对应的精确解值

    # 转换为稀疏矩阵的压缩行格式
    A = A.tocsr()

    # 求解线性方程组
    u = spsolve(A, b)

    # 将一维数组转换为二维数组
    u = u.reshape((n, n))

    # 调用 mesh_grid 函数获取网格的坐标
    X, Y = mesh_grid(n)

    # 绘制三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    plt.title('Solution to Poisson Equation')
    plt.show()
    return u


# 创建一个n=8的网格，并求解泊松方程

u = solve_poisson_equation(0.2)
print(u)
# 创建图像
# create_plot(n, u)







