import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def mesh_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


def generate_triangle_elements(X, Y):

    # 获取网格点的数量
    num_points = X.size

    # 将二维网格点转换为一维数组
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # 生成三角形的连接信息
    # 假设我们有一个规则的网格，每个方形由两个三角形组成
    triangles = np.zeros((2 * (N - 1) ** 2, 3), dtype=int)
    index = 0
    for i in range(N - 1):
        for j in range(N - 1):
            # 第一个三角形
            triangles[index] = [i * N + j, i * N + j + 1, (i + 1) * N + j]
            index += 1
            # 第二个三角形
            triangles[index] = [i * N + j + 1, (i + 1) * N + j + 1, (i + 1) * N + j]
            index += 1
    return triangles


# 生成网格点
N = 4
X, Y = mesh_grid(N)

# 生成三角形元素
triangles = generate_triangle_elements(X, Y)

# 绘制三角形元素
plt.figure()
plt.triplot(X.flatten(), Y.flatten(), triangles, 'bo-', linewidth=1)
plt.title('Triangular Elements from Mesh Grid')
plt.show()


