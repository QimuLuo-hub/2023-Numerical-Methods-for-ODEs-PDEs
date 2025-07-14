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
def ForwardEuler(x0, x1, h, y0):
    n = round((x1-x0)/h) #compute n values
    x = np.linspace(x0, x1, n+1)#get each x
    x[0] = x0#set initial value of x can not be 0 'float' object cannot be interpreted as an integer

    y = np.zeros(n + 1)#set zero array to save Numerical y
    y[0] = rhs(x0, y0)#set initial value of y

    error = np.zeros(n+1)
    for i in range(n):
        y[i+1] = y[i]+ h * rhs(x[i], y[i])#rhs(x[i], Numerical_y[i]) instead of Numerical_y[i] directly
        error[i+1] = abs(y[i+1]-exact_solution(x[i+1]))
    error = np.max(error)

    plt.plot(x, y, '-o', label='Numerical Solution')
    plt.plot(x, exact_solution(x), label='Exact Solution')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.title('Comparison of Forward Euler Method')

    fig = plt.gcf()
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
    filename = 'Comparison_of_Forwardeuler.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()
    return y, error

# plug in value, call function

ForwardEuler(0, 100, 0.2, 1)