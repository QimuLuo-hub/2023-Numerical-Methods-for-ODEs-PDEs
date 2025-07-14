import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

'''
Points: Vertice coordinate and global index
Nx = number of elemnts points in x direction
Ny = number of elemnts points in y direction
'''
n = 4


def rhs(x, y):
    return 2 * (x + y - x * x - y * y)

def exactsol(x, y):
    return x * y * (1 - x) * (1 - y)


def mesh_grid(i, n):
    x_axis = np.linspace(0, 1, n + 1, endpoint=True)
    y_axis = np.linspace(0, 1, n + 1, endpoint=True)
    # xx, yy = np.meshgrid(x_axis, y_axis)
    yy, xx = np.meshgrid(y_axis, x_axis)
    xy = np.stack([xx, yy], axis=2)
    vertice = xy[i[0], i[1]]
    # 定位所有网格点 并获取顶点的 x 坐标和 y 坐标， i 相当于是全局坐标idx传入，如果是三角形的话 三个顶点的全局坐标idx有一定的关联关系
    # index_1d = i[0] * (n + 1) + i[1]
    index_1d = i[1] * (n + 1) + i[0]

    return vertice, index_1d


class solve_poisson_equation_cst:
    def __init__(self, vertice_i, veritice_j, veritice_k):
        self.vertice_i = vertice_i
        self.vertice_j = veritice_j
        self.vertice_k = veritice_k

        self.vertice_i, self.index_i = mesh_grid(self.vertice_i, n)
        self.vertice_j, self.index_j = mesh_grid(self.vertice_j, n)
        self.vertice_k, self.index_k = mesh_grid(self.vertice_k, n)

        self.x1, self.y1 = self.vertice_i[0], self.vertice_i[1]
        self.x2, self.y2 = self.vertice_j[0], self.vertice_j[1]
        self.x3, self.y3 = self.vertice_k[0], self.vertice_k[1]

        self.area = 0.5 * np.abs(
            self.x1 * (self.y2 - self.y3) + self.x2 * (self.y3 - self.y1) + self.x3 * (self.y1 - self.y2))

    '''Function to modify'''

    # def rhs(self, x, y):
    #     return 2 * (x + y - x * x - y * y)
    #     # raise NotImplementedError
    # def exactsol(self, x, y):
    #     return x * y * (1-x) * (1-y)
    # raise NotImplementedError

    def stiffness_matrix(self, global_K):
        diff_ij = self.vertice_i - self.vertice_j
        diff_ik = self.vertice_i - self.vertice_k
        diff_jk = self.vertice_j - self.vertice_k

        #local_K = np.zeros((3, 3))
        local_K = np.array(
            [[np.dot(diff_jk, diff_jk), np.dot(diff_ik, -diff_jk), np.dot(diff_ij, diff_jk)],
             [np.dot(diff_ik, -diff_jk), np.dot(diff_ik, diff_ik), np.dot(-diff_ij, diff_ik)],
             [np.dot(diff_ij, diff_jk), np.dot(-diff_ij, diff_ik), np.dot(diff_ij, diff_ij)]
             ]) / (4 * self.area)
        # print('local_K[3]', local_K[2])
        # plt.imshow(local_K)
        # plt.show()
        sub_index = [self.index_i, self.index_j, self.index_k]

        global_K[np.ix_(sub_index,
                        sub_index)] += local_K  # 用np.ix_(行， 列)来更新矩阵的元素，因为三角形三个顶点的全局坐标已经覆盖了全部行(Nx+1) * (Ny+1) 以及全部列

        # print('global_K', global_K)
        # plt.imshow(global_K)
        # plt.show()
        # return global_K

    def load_vector(self, global_F):
        def numerical_integration(x, y):
            shape_i = (self.x2 * self.y3 - self.x3 * self.y2 + x * (self.y2 - self.y3) + y * (self.x3 - self.x2)) / (
                        2 * self.area)
            shape_j = (self.x3 * self.y1 - self.x1 * self.y3 + x * (self.y3 - self.y1) + y * (self.x1 - self.x3)) / (
                        2 * self.area)
            shape_k = (self.x1 * self.y2 - self.x2 * self.y1 + x * (self.y1 - self.y2) + y * (self.x2 - self.x1)) / (
                        2 * self.area)

            shape_function = [shape_i, shape_j, shape_k]
            local_F = [shape_function[i] * rhs(x, y) * self.area for i in range(3)]
            local_F = np.array(local_F)
            return local_F

        x = (self.x1 + self.x2 + self.x3) / 3
        y = (self.y1 + self.y2 + self.y3) / 3

        local_F = numerical_integration(x, y)

        global_F[self.index_i] += local_F[0]
        global_F[self.index_j] += local_F[1]
        global_F[self.index_k] += local_F[2]

        return local_F[0], local_F[1], local_F[2]


element_list = []
# 对于每个内部格点，需要添加两个三角形
for ix in range(n):
    for iy in range(n):
        element_list.append(solve_poisson_equation_cst((ix, iy), (ix + 1, iy), (ix, iy + 1)))
        element_list.append(solve_poisson_equation_cst((ix, iy + 1), (ix + 1, iy), (ix + 1, iy + 1)))

global_stiffness = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
#global_stiffness = lil_matrix(((n + 1) ** 2, (n + 1) **2 ))
global_load = np.zeros((n + 1) ** 2)

'''stiffness'''
[ele.stiffness_matrix(global_stiffness) for ele in element_list]
# print(global_stiffness)
# global_stiffness = global_stiffness.tocsr()

'''Load vector'''
[ele.load_vector(global_load) for ele in element_list]
# print('global_load_before',  global_load)
# print('global_load_before_shape',  global_load.shape)

# '''boundary conditions'''
# x = np.linspace(0, 1, n)
# h = x[1] - x[0]
# for i in range(n + 1):
#     global_stiffness[i * (n + 1), i * (n + 1)] = 1  # left
#     global_load[i * (n + 1)] = exactsol(0, i * h)  # fill the left boundary condition

#     global_stiffness[i * (n + 1) + n, i * (n + 1) + n] = 1  # right
#     global_load[i * (n + 1) + n] = exactsol(1, i * h)

#     global_stiffness[i, i] = 1  # down
#     global_load[i] = exactsol(i * h, 0)

#     global_stiffness[(n + 1) * n + i, (n + 1) * n + i] = 1  # up
#     global_load[(n + 1) * n + i] = exactsol(i * h, 1)

# numerical_u = spsolve(global_stiffness, global_load)
# print(numerical_u)
# print('global_load_afterboundary',  global_load)
# print('global_load_after_shape',  global_load.shape)


numerical_u = np.linalg.solve(global_stiffness, global_load)
# print('numerical_u', numerical_u)
numerical_u = numerical_u.reshape((n+1, n+1))

# exact_u =  [exactsol(i*h, i*h) for i in range(n + 1)]
# error = abs(numerical_u-exact_u)
# print('numerical_u', numerical_u)
# print(numerical_u.shape)

def visualize(n):

    '''Numerical solution'''
    x = np.linspace(0, 1, n + 1, endpoint=True)
    y = np.linspace(0, 1, n + 1, endpoint=True)
    Y, X = np.meshgrid(y, x)
    # X, Y = np.meshgrid(x, y)
    u_real = exactsol(X, Y)
    # error = np.zeros(numerical_u.shape)
    # error[:] = abs(numerical_u - u_real)
    # error_max = np.max(error)

    '''Visualize'''
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, numerical_u, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u')
    plt.title('Numerical ')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_real, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u')
    plt.title('Analytical')

    # ax3 = fig.add_subplot(133, projection='3d')
    # ax3.plot_surface(X, Y, error, cmap='viridis')
    # ax3.set_xlabel('X')
    # ax3.set_ylabel('Y')
    # ax3.set_zlabel('Z')
    # ax3.set_title("Error")

    # filename = (f'Five-Points_Difference_Numerical_Analytical_at h = {h}.pdf')
    # plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

    # return error_max

# error_max = visualize(n)
# print(error_max)
visualize(n)






