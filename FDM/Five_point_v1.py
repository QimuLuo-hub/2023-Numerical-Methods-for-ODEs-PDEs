import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def exact_sol(x, y):
    # output = np.sin(np.pi * x) * np.sin(np.pi * y)
    output = np.sin(x) * np.sin(y)
    return output

def rhs_f(x , y):
    # output = -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    output = 2 * np.sin(x) * np.sin(y)
    return output

def mesh_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    [X, Y] = np.meshgrid(x, y)
    return X, Y

def solve_poisson_equation(x0, x1, h):
    n = round((x1-x0)/h) #number of points in x and y direction

    # construct sparse coefficient matrix A and initialize b
    A = lil_matrix((n ** 2, n ** 2))
    b = np.zeros(n ** 2)

    # Coefficients For inner points
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            index_inner = i * n + j

            #main diagonal
            A[index_inner, index_inner] = -4 / h ** 2

            # neighbourhood
            A[index_inner, (i - 1) * n + j] = 1 / h ** 2  #down
            A[index_inner, (i + 1) * n + j] = 1 / h ** 2  #up
            A[index_inner, index_inner - 1] = 1 / h ** 2  #left
            A[index_inner, index_inner + 1] = 1 / h ** 2  #right

            # b matrix for rhs function (inner)
            b[i * n + j] = rhs_f(i * h, j * h)  # fill use coordinates

    # set boundary conditions in matrix A: set 1 -- fix the boundary
    for i in range(n):
        A[i * n, i * n] = 1 #left
        b[i * n] = exact_sol(0, i * h)  # fill the left boundary condition

        A[i * n + n - 1, i * n + n - 1] = 1  # right
        b[i * n + n - 1] = exact_sol(1, i * h)

        A[i, i] = 1  #down
        b[i] = exact_sol(i * h, 0)

        A[(n - 1) * n + i, (n - 1) * n + i] = 1  # up
        b[(n - 1) * n + i] = exact_sol(i * h, 1)

    # save the sparse matrix as the csr form
    A = A.tocsr()

    # obtain the solution u
    u = spsolve(A, b)

    # reshape the solution (already achieve dimension-deduction)
    u = u.reshape((n, n))

    print(u.shape)
    # u = u.reshape((n))
    '''Error'''
    X, Y = mesh_grid(n)
    u_real = exact_sol(X, Y)
    error = np.zeros((n, n))
    error[:] = abs(u-u_real)
    error_max = np.max(error)

    '''Visualize'''
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, u, cmap='viridis')
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
    # ax3.plot_surface(X, Y, u, cmap='Blues')  # blue for mesh
    # ax3.plot_surface(X, Y, u_real, cmap='Reds')  # red for real
    # ax3.set_xlabel('X')
    # ax3.set_ylabel('Y')
    # ax3.set_zlabel('Z')
    # ax3.set_title("Comparison")

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap='viridis')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("Error")

    filename = (f'Five-Points_Difference_Numerical_Analytical_at h = {h}.pdf')
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()

    return error_max

x0 = 0
x1 = 1
h = 0.1
u = solve_poisson_equation(x0, x1, h)



