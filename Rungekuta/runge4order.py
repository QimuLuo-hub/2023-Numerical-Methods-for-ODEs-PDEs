import math
import matplotlib.pyplot as plt
import numpy as np

'''Set numerical solution and Analytical solution'''
def rhs(x, y):
    output = y
    return output

def exact_solution(x):
    output = np.exp(x)
    return output

'''obtain max error at a specific step and numerical y '''
def rungekutta4(x0, x1, h, y0):
    n = round((x1-x0)/h) #compute n values
    x = np.linspace(x0, x1, n+1)#get each x
    x[0] = x0#set initial value of x can not be 0 'float' object cannot be interpreted as an integer
    y = np.zeros(n + 1)#set zero array to save Numerical y
    y[0] = rhs(x0, y0)#set initial value of y

    error = np.zeros(n+1)
    for i in range(n):
        h = x[i+1] - x[i]
        '''4 order: '''
        k1 = rhs(x[i], y[i])
        k2 = rhs(x[i] + 1/2*h, y[i] + 1/2*h*k1)
        k3 = rhs(x[i] + 1/2*h, y[i] + 1/2*h*k2)
        k4 = rhs(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        '''error'''
        error[i+1] = abs(y[i+1]-exact_solution(x[i+1]))
    error = np.max(error)
    plt.plot(x, y, '-o', label='Numerical Solution')
    plt.plot(x, exact_solution(x), label='Exact Solution')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.title('Comparison of rungekutta4 Method')

    fig = plt.gcf()
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
    filename = 'Comparison_of_rungekutta4.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()
    return y, error

# plug in value, call function

rungekutta4(0, 1, 0.2, 1)