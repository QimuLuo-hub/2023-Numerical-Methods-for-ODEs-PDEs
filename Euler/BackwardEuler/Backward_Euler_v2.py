import matplotlib.pyplot as plt
import numpy as np

'''use t to replace y later'''
# define f(t,y)
def rhs(t, y):
    output = np.sin(t)*y
    return output

# define fprime: w.r.t. y!!!
def rhs_prime(t):
    output = np.sin(t)
    return output

def exactsol(t):
    output = np.exp(-np.cos(t))
    return output


'''Three criterion for Newton method: rate、differential、function'''

def rate_newton(g, gprime, x0, epsilon, max_interation):
    for _ in range(max_interation):
        x1 = x0 - g(x0) / gprime(x0) #x0 -y_n; x1-y_n+1
        if abs(x1-x0)/x0 < epsilon:
            return x1
        else:
            x0 = x1
    return None
    # Return None if abs(x1 - x0) / x0 has not < epsilon within max_interation

# def differentialnewton(g, gprime, x0, epsilon, max_interation):
#     for i in range(max_interation):
#         x1 = x0 - g(x0) / gprime(x0)
#         if abs(x1 - x0) < epsilon:
#             return x1
#         else:
#             x0 = x1
#     return None
#
#
# def funcctionnewton(g, gprime, x0, epsilon, max_interation):
#     for j in range(max_interation):
#         x1 = x0 - g(x0) / gprime(x0)
#         if abs(g(x0)) < epsilon:
#             return x1
#         else:
#             x0 = x1
#     return None


'''Backward Euler Method'''
def backward_euler(t0, tf, h, y0, epsilon, max_interation):
    n = round((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    t[0] = t0

    y = np.zeros(n + 1)
    y[0] = y0

    Error = np.zeros(n + 1)
    for i in range(n):
        '''
        Newton interation-
        use anonymous function x to compute y_n+1 and obtain gprime conveniently
        '''
        g = lambda x: -x + y[i] + h * rhs(t[i + 1], x)
        x0 = y[i]  # replacement by anonymous variable
        gprime = lambda x: -1 + h * rhs_prime(t[i + 1])

        '''numerical_y'''
        y[i + 1] = rate_newton(g, gprime, x0, epsilon, max_interation)#x1
        Error[i + 1] = abs(y[i + 1] - exactsol(t[i + 1]))

    error = np.max(Error)

    # plt.plot(t, y, '-o', label='Numerical Solution')
    # plt.plot(t, exactsol(t), label='Exact Solution')
    #
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.legend(loc='best')
    # plt.title('Comparison of Backward Euler Method')
    #
    # fig = plt.gcf()
    # fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
    # filename = 'Comparison_of_Backward_Euler.pdf'
    # plt.savefig(filename, format='pdf', bbox_inches='tight')
    # plt.show()
    return y, error

# plug in value
t0 = 0
tf = 1
h = 0.2
y_0 = np.exp(-1)
epsilon = 0.01
max_interation = 100

backward_euler(t0, tf, h, y_0, epsilon, max_interation)
