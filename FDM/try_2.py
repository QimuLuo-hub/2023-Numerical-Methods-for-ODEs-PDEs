import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import math
import matplotlib.pyplot as plt


def mesh_grid(N):
    x = np.linspace(0, 1, N+1 )
    y = np.linspace(0, 1, N+1)
    [X, Y] = np.meshgrid(x, y)
    return X, Y

# def rhs_f(x , y):
#     output = -2 * math.pi ** 2 * np.sin(math.pi * x) * np.sin(math.pi * y)
#     return output
def rhs_f(x , y):
    # 仅计算内部点的值
    output = -2 * math.pi ** 2 * np.sin(math.pi * x[1:-1, 1:-1]) * np.sin(math.pi * y[1:-1, 1:-1])
    return output


def exact_sol(x, y):
    output = np.sin(math.pi * x) * np.sin(math.pi * y)
    return output

# n is the unit  4
def poisson_solver_sparse(n):
    h = 1 / n  # 步长
    X, Y = mesh_grid(n)

    # 构建系数矩阵A
    main_diag = -4 * np.ones(n-1)
    off_diag = 1 * np.ones(n-1)
    diagonals = [main_diag, off_diag, off_diag]
    offsets = [0, -1, 1]
    A = diags(diagonals, offsets, shape=(n-1, n-1), format='csr')

    # 构建右侧项 向量b
    b = np.zeros((n - 1, n - 1))
    # 填充内部
    b[1:-1, 1:-1] = rhs_f(X, Y)[1:-1, 1:-1]

    # 设置边界条件
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)

    b[:, 0] = exact_sol(x[0], y[1:-1])  # 左边界
    b[:, -1] = exact_sol(x[-1], y[1:-1])  # 右边界
    b[0, :] = exact_sol(x[1:-1], y[0])  # 上边界
    b[-1, :] = exact_sol(x[1:-1], y[-1])  # 下边界


    # 将边界条件乘以步长的平方
    b = b * h ** 2

    u = spsolve(A, b)

    # 将解重塑为二维数组
    u = u.reshape((n-1, n-1))

    u_real = exact_sol(X, Y)[1:-1, 1:-1]
    error = abs(u-u_real)

    # 绘制解的三维图
    fig = plt.figure(figsize=(12, 5))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot_surface(X[1:-1, 1:-1], Y[1:-1, 1:-1], u, cmap='viridis')
    # ax.set_title('Solution')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('u')

    # 绘制解析解和误差解的对比
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X[1:-1, 1:-1], Y[1:-1, 1:-1], u_real,alpha=0.5)
    ax.plot_surface(X[1:-1, 1:-1], Y[1:-1, 1:-1], u, alpha=0.5)
    ax.set_title('Exact Solution vs. Numerical Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    # ax.legend(['Exact Solution', 'Numerical Solution'])

    plt.show()

    return error
print(poisson_solver_sparse(50))

    # # 设置边界条件
    # b[:, 0] += 1
    # b[:, -1] += 1
    # b[0, :] += 1
    # b[-1, :] += 1

# b[:, 0] = exact_sol(X[:, 0], Y[:, 0])
# b[:, -1] = exact_sol(X[:, -1], Y[:, -1])
# b[0, :] = exact_sol(X[0, :], Y[0, :])
# b[-1, :] = exact_sol(X[-1, :], Y[-1, :])

# x = np.linspace(0, 1, n - 1)
# y = np.linspace(0, 1, n - 1)
#
# # 设置边界条件
# b[:, 0] = exact_sol(0, y * h)
# b[:, -1] = exact_sol(1, y * h)
# b[0, :] = exact_sol(x * h, 1)
# b[-1, :] = exact_sol(x * h, 0)