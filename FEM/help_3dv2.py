import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Nx = 4
Ny = 4

class Points:
    x_space = np.linspace(0, 1, Nx + 1, endpoint=True)
    y_space = np.linspace(0, 1, Ny + 1, endpoint=True)
    yy, xx = np.meshgrid(y_space, x_space)
    xy = np.stack([xx, yy], axis=2)
    # print('xx:', xx)
    # print('xx[1:-1, 1:-1]', xx[1:-1, 1:-1])
    # print('yy', yy)
    # print('yy[1:-1, 1:-1]', yy[1:-1, 1:-1])
    # print('xy[1,0]', xy[1, 0])
    # print(yy)
    @staticmethod
    def get(i):
        return Points.xy[i[0], i[1]]

    @staticmethod
    def global_idx(i):
        return i[1] * (Nx + 1) + i[0] #结果没有差，这个更符合我的习惯 从左下角开始向右逆时针标号
        # return i[0] * (Nx + 1) + i[1]

class TriangleElement:

    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    @property
    def p1(self):
        return Points.get(self.i)

    @property
    def p2(self):
        return Points.get(self.j)

    @property
    def p3(self):
        return Points.get(self.k)

    @property
    def area(self):
        # 既然是均匀网格，可以简单地
        # return 1 / Nx / Ny /2
        return (self.p2[0] * self.p3[1] - self.p2[1] * self.p3[0] +
                self.p3[0] * self.p1[1] - self.p3[1] * self.p1[0] +
                self.p1[0] * self.p2[1] - self.p1[1] * self.p2[0]) / 2
       

    def add_local_stiffness(self, global_stiffness: np.array):
        gi = Points.global_idx(self.i)
        gj = Points.global_idx(self.j)
        gk = Points.global_idx(self.k)
        sub_idx = [gi, gj, gk]#一个element 的三个顶点的全局坐标
        # i, j, k = Points.get(self.i), Points.get(self.j), Points.get(self.k)
        i, j, k = self.p1, self.p2, self.p3
        local_stiff = np.sum(np.array(
            [[(j - k) ** 2, (i - k) * (k - j), (i - j) * (j - k)],
             [(i - k) * (k - j), (i - k) ** 2, (k - i) * (i - j)],
             [(i - j) * (j - k), (k - i) * (i - j), (i - j) ** 2]]), axis=2) / (4 * self.area)
        
        
        global_stiffness[np.ix_(sub_idx, sub_idx)] += local_stiff

        #全局刚度矩阵的行和列都是sub_idx
        # sub_indx 以三角形顶点的全局坐标的方式遍历了所有行和所有列 共 Nx+1 * Ny + 1 个维度
        # print('shape_local', local_stiff.shape)
        # print('shape_global',global_stiffness.shape)
        # plt.imshow(global_stiffness)
        # plt.show()

    '''Load vector'''
    def add_local_load(self, f: callable, global_load: np.array):
        gi = Points.global_idx(self.i)
        gj = Points.global_idx(self.j)
        gk = Points.global_idx(self.k)
        i, j, k = Points.get(self.i), Points.get(self.j), Points.get(self.k)

        def compose_fun(local_idx):
            if local_idx == 0:
                li, lj, lk = i, j, k
            if local_idx == 1:
                lk, li, lj = i, j, k
            if local_idx == 2:
                lj, lk, li = i, j, k

            def _fun(x, y):
                _r = ( x * (lj[1] - lk[1]) +
                       y * (lk[0] - lj[0]) +
                       lj[0] * lk[1] - lk[0] * lj[1]) / (2 * self.area)
                return _r * f(x, y)

            return _fun

        global_load[gi] += self.quadrature(compose_fun(0))
        global_load[gj] += self.quadrature(compose_fun(1))
        global_load[gk] += self.quadrature(compose_fun(2))

        return self.quadrature(compose_fun(0)), self.quadrature(compose_fun(1)), self.quadrature(compose_fun(2))

    def quadrature(self, q: callable):
        return q((self.p1[0] + self.p2[0] + self.p3[0]) / 3, (self.p1[1] + self.p2[1] + self.p3[1]) / 3) * self.area

        # return (
        #         q((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2) 
        #         + q((self.p2[0] + self.p3[0]) / 2, (self.p2[1] + self.p3[1]) / 2)
        #          + q((self.p3[0] + self.p1[0]) / 2, (self.p3[1] + self.p1[1]) / 2)
        # ) / 3 * self.area

# def q(xx, yy):
#     return 2 * (xx + yy - xx * xx - yy * yy)
#
# element_list = []
# for ix in range(Nx):
#     for iy in range(Ny):
#         element_list.append(TriangleElement((ix, iy), (ix + 1, iy), (ix, iy + 1)))
#         element_list.append(TriangleElement((ix, iy + 1), (ix + 1, iy), (ix + 1, iy + 1)))
# global_stiffness = np.zeros(((Nx + 1) * (Ny + 1), (Nx + 1) * (Ny + 1)))
# [te.add_local_stiffness(global_stiffness) for te in element_list]
# interior_idx = (Nx + 1) * np.arange(1, Nx).reshape(Nx - 1, 1) + np.arange(1, Ny).reshape(1, Ny - 1)
# interior_idx = interior_idx.ravel()
# global_stiffness = global_stiffness[np.ix_(interior_idx, interior_idx)]
# global_load = np.zeros((Nx + 1) * (Ny + 1))
# [te.add_local_load(q, global_load) for te in element_list]
# global_load = global_load[interior_idx]
# res = np.linalg.solve(global_stiffness, global_load)
# # print(res)
# res = res.reshape(Nx - 1, Ny - 1)
# res = np.pad(res, pad_width=1, mode='constant', constant_values=0)
# plt.pcolormesh(Points.xx, Points.yy, res)
# plt.show()

'''RHS function'''
def q(xx, yy):
    # return 2 * np.sin(xx) * np.sin(yy)
    # return 2 * np.pi ** 2 * np.sin(np.pi * xx) * np.sin(np.pi * yy)
    return 2 * (xx + yy - xx * xx - yy * yy)

def true_solution(xx, yy):
    # return np.sin(np.pi * xx) * np.sin(np.pi * yy)
    # return np.sin(xx) * np.sin(yy)
    return xx * yy * (1-xx) * (1-yy)


element_list = []
#对于每个内部格点，需要添加两个三角形
for ix in range(Nx):
    for iy in range(Ny):
        element_list.append(TriangleElement((ix, iy), (ix + 1, iy), (ix, iy + 1)))
        element_list.append(TriangleElement((ix, iy + 1), (ix + 1, iy), (ix + 1, iy + 1)))

# print('element_list', element_list)


'''stiffness'''
global_stiffness = np.zeros(((Nx + 1) * (Ny + 1), (Nx + 1) * (Ny + 1)))
[te.add_local_stiffness(global_stiffness) for te in element_list]
# print('global_stiffness', global_stiffness)

'''load'''
interior_idx = (Nx + 1) * np.arange(1, Nx).reshape(Nx - 1, 1) + np.arange(1, Ny).reshape(1, Ny - 1)
interior_idx = interior_idx.ravel()

global_stiffness = global_stiffness[np.ix_(interior_idx, interior_idx)] # inner


global_load = np.zeros((Nx + 1) * (Ny + 1))
[te.add_local_load(q, global_load) for te in element_list]
print('global_load', global_load)

# global_load = global_load[interior_idx] #global
numerical_u = np.linalg.solve(global_stiffness, global_load)

print('numerical_u', numerical_u )

# 计算真解在网格点上的值
# true_solution_values = true_solution(Points.xx[1:-1, 1:-1], Points.yy[1:-1, 1:-1])
true_solution_values = true_solution(Points.xx[:-2, :-2], Points.yy[:-2, :-2])
# true_solution_values = true_solution(Points.xx, Points.yy)
# 计算数值解在网格点上的值
numerical_solution_values = numerical_u.reshape(Nx - 1, Ny - 1)

# 计算L2 误差
l2_error = np.linalg.norm(true_solution_values - numerical_solution_values, ord=2)
l_infty_error = np.max(np.abs(true_solution_values - numerical_solution_values))

print("L2 Error:", l2_error)

print("L∞ Error:", l_infty_error)


# res = res.reshape(Nx - 1, Ny - 1)