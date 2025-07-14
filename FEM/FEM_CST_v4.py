import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

'''
Points: vertices coordinate and global index
n = number of elements  in x or y direction
'''
n = 50

def rhs(x, y):
    # output = 2 * (x + y - x * x - y * y)
    output = 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    return output

def exactsol(x, y):
    # output = x * y * (1 - x) * (1 - y)
    output = np.sin(np.pi * x) * np.sin(np.pi * y)
    return output

# N = 100
# fig = plt.figure(figsize=(15, 5))
# def visualize_exact(N):
#     xx = np.linspace(0, 1, N + 1)
#     yy = np.linspace(0, 1, N + 1)
#     [X, Y] = np.meshgrid(xx, yy)
#     u_real = exactsol(X, Y)
#     ax2 = fig.add_subplot(132, projection='3d')
#     ax2.plot_surface(X, Y, u_real, cmap='viridis')
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
#     ax2.set_zlabel('u')
#     plt.title('Analytical')
#     return plt.show()

# visualize_exact(N)  



def mesh_grid(i, n):
    x_axis = np.linspace(0, 1, n + 1, endpoint=True)
    y_axis = np.linspace(0, 1, n + 1, endpoint=True)
    yy, xx = np.meshgrid(y_axis, x_axis)
    xy = np.stack([xx, yy], axis=2) #connect x and y coordinates
    vertices = xy[i[0], i[1]]

    '''
    Locate the coordinate of vertices via 1d_global_index, for triangular elements, 
    one grid is divided into 2 elements
    ix, iy: from 0 to n , anticlockwise
    (ix, iy), (ix + 1, iy), (ix, iy + 1)))
    (ix, iy + 1), (ix + 1, iy), (ix + 1, iy + 1))
    '''

    index_1d = i[1] * (n + 1) + i[0] # global_index from the left corner
    return vertices, index_1d

class solve_poisson_equation_cst:
    def __init__(self, vertices_i, vertices_j, vertices_k):
        self.vertices_i = vertices_i
        self.vertices_j = vertices_j
        self.vertices_k = vertices_k

        self.vertices_i, self.index_i = mesh_grid(self.vertices_i, n)
        self.vertices_j, self.index_j = mesh_grid(self.vertices_j, n)
        self.vertices_k, self.index_k = mesh_grid(self.vertices_k, n)

        self.x1, self.y1 = self.vertices_i[0], self.vertices_i[1]
        self.x2, self.y2 = self.vertices_j[0], self.vertices_j[1]
        self.x3, self.y3 = self.vertices_k[0], self.vertices_k[1]

        det = np.linalg.det(np.array([
        [self.x1, self.y1, 1],
        [self.x2, self.y2, 1],
        [self.x3, self.y3, 1]])) 

        self.area = 0.5 * np.abs(det) 

    '''   
    KU = F
    ∫\nabla phi_i * \nabla phi_j dV * u_k = ∫ phi_i * f dV
    '''
    # LHS: stiffness_matrix can be computed explicitly
    def stiffness_matrix(self, global_K):
        diff_ij = self.vertices_i - self.vertices_j
        diff_ik = self.vertices_i - self.vertices_k
        diff_jk = self.vertices_j - self.vertices_k

        local_K = np.array(
            [[np.dot(diff_jk, diff_jk), np.dot(diff_ik, -diff_jk), np.dot(diff_ij, diff_jk)],
             [np.dot(diff_ik, -diff_jk), np.dot(diff_ik, diff_ik), np.dot(-diff_ij, diff_ik)],
             [np.dot(diff_ij, diff_jk), np.dot(-diff_ij, diff_ik), np.dot(diff_ij, diff_ij)]
             ]) / (4 * self.area)
        
        sub_index = [self.index_i, self.index_j, self.index_k]#obtain the 1d_index coordinates for global_K

        global_K[np.ix_(sub_index, sub_index)] += local_K 
        #np.ix_() return an array (locate rows, locate columns)
        #the global_index for vertices have covered all the rows and columns

    #RHS:load_vector apply numerical integral (QUADRATURE_RULES_TRI)
    #link:mhttps://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html

    def load_vector(self, global_F):
        def numerical_integration(x, y):
            shape_i = (self.x2 * self.y3 - self.x3 * self.y3 + x * (self.y2 - self.y2) + y * (self.x3 - self.x2)) / (2 * self.area)
            shape_j = (self.x3 * self.y1 - self.x1 * self.y3 + x * (self.y3 - self.y1) + y * (self.x1 - self.x3)) / (2 * self.area)
            shape_k = (self.x1 * self.y2 - self.x2 * self.y1 + x * (self.y1 - self.y2) + y * (self.x2 - self.x1)) / (2 * self.area)
            shape_function = [shape_i, shape_j, shape_k]

            local_F = np.array([shape_function[i] * rhs(x, y) * self.area for i in range(3)])
            return local_F

        x = (self.x1 + self.x2 + self.x3) / 3
        y = (self.y1 + self.y2 + self.y3) / 3

        local_F = numerical_integration(x, y)

        global_F[self.index_i] += local_F[0]
        global_F[self.index_j] += local_F[1]
        global_F[self.index_k] += local_F[2]


'''
Populate the global_index for elements (1 grid - 2 elements)
'''
start_time = time.time()
element_list = []
for ix in range(n):
    for iy in range(n):
        element_list.append(solve_poisson_equation_cst((ix, iy), (ix + 1, iy), (ix, iy + 1)))
        element_list.append(solve_poisson_equation_cst((ix, iy + 1), (ix + 1, iy), (ix + 1, iy + 1)))

global_stiffness = np.zeros(((n + 1) * (n + 1), (n + 1) * (n + 1)))
global_load = np.zeros((n + 1) * (n + 1))


#Inner index
index_inner = []
for i_inner in range(1, n):
    for j_inner in range(1, n):
        index = i_inner * (n + 1) + j_inner
        index_inner.append(index)
# print(index_inner)

'''stiffness'''
[elements.stiffness_matrix(global_stiffness) for elements in element_list]
global_stiffness = global_stiffness[np.ix_(index_inner, index_inner)]
# print('global_stiffness', global_stiffness)
# print('global_stiffness.shape', global_stiffness.shape)

'''Load vector'''
[elements.load_vector(global_load) for elements in element_list]
global_load = global_load[index_inner]
# print('global_load', global_load)
# print('global_load.shape', global_load.shape)

'''Solve U'''
numerical_u = np.zeros((n + 1) * (n + 1))
numerical_u_inner = np.linalg.solve(global_stiffness, global_load)


'''Add boundary conditions'''
numerical_u[index_inner] = numerical_u_inner

x = np.linspace(0, 1, n)
h = x[1] - x[0]

for i in range(n + 1):
    numerical_u[i * (n + 1)] = exactsol(0, i * h)  # left

    numerical_u[i * (n + 1) + n] = exactsol(1, i * h)  # right

    numerical_u[i] = exactsol(i * h, 0)  # down

    numerical_u[(n + 1) * n + i] = exactsol(i * h, 1)  # up

numerical_u = numerical_u.reshape((n + 1, n + 1))


def visualize(n):
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    [X, Y] = np.meshgrid(x, y)

    u_real = exactsol(X, Y)
    error = np.zeros((n + 1, n + 1))
    error[:] = abs(numerical_u - u_real)
    L_infinity = np.max(error)
    l2_error = np.linalg.norm(u_real - numerical_u, ord=2)


    '''Visualize'''
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('With Boundary Conditions', fontsize=16, fontweight='bold')
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, numerical_u, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u')
    plt.title('Numerical')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_real, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u')
    plt.title('Analytical')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap='viridis')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("Error")

    filename = (f'FEM_Numerical_Analytica_BCs_at n = {n}.pdf')
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

    return L_infinity, l2_error

L_infinity, l2_error = visualize(n)
end_time = time.time()
run_time = end_time - start_time
print('at n = ', n, 'L_infinity', L_infinity, 'l2_error', l2_error, 'run time', run_time )




