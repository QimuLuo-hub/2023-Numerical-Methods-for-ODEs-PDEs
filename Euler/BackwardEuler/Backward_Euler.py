import math
import matplotlib.pyplot as plt
import numpy as np

def rhs_f(x, y):
    output = np.sin(x)*y
    return output

def f_prime(t):
    prime = np.sin(t)
    return prime

def exactsol(x, y):
    output = np.exp(-np.cos(x))
    return output


# def newton_interation(t0, epsilon, max_inte):
#     t = t0
#     for i in range(max_inte):
#         t[i+1] = t[i]-
def backwardEuler(x0, x1, h, y0, epsilon):
    n = round((x1-x0)/h)
    x = np.linspace(x0, x1, n+1)
    x[0] = x0

    y = np.zeros(n+1)
    y[0] = y0

    error = np.zeros(n+1)
    for i in range(n):
        h = x[i + 1] - x[i]
        for k in range(10000):
            if y[k+1]-y[k] < epsilon:
                y[k+1] = y[k] - (rhs_f(x[k+1], y[k+1]))/(f_prime(y[k+1]))
            else:
                break
        y = y[k + 1]
        y = y[i]+h*rhs_f(x[i+1], y[i+1])
    error = np.max(error)
    return y, error


# plug in value, call function

backwardEuler(0, 1, 0.2, 1, 0.01)
# plt.plot(x, y, '-o', label='Numerical Solution')
#     plt.plot(x, exactsol(x, y), label='Exact Solution')
#
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(loc='best')
#     plt.title('Comparison of Backward Euler Method')
#
#     fig = plt.gcf()
#     fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
#     filename = 'Comparison_of_Backwardeuler.jpg'
#     plt.savefig(filename, format='jpg', bbox_inches='tight')
#     plt.show()







