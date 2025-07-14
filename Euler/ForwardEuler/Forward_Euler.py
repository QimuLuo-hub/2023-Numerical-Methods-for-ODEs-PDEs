import numpy as np
import matplotlib.pyplot as plt

'''Solve pde use Euler method'''
n = 5
x = np.linspace(0, 1, n + 1)
h = 0.2

def f(x, y):
    output = y
    return output

def exactsol(x):
    y = np.exp(x)
    return y
T_y = np.array([exactsol(x) for x in x])

N_y = np.zeros(n+1)
N_y[0] = f(0, 1)
for i in range(1, n + 1):
    N_y[i] = N_y[i - 1] + h * N_y[i - 1]
    print('interation', i, N_y[i])

'''Figure 1: show Difference'''
fig, ax = plt.subplots()
ax.plot(x, N_y, '-o', label = 'Numerical solution')
ax.plot(x, T_y, label = 'Exact solution', linewidth=2)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()




















