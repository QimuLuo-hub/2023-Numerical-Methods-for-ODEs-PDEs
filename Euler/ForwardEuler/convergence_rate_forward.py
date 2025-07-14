import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd

n_values = [10**(i+1) for i in range(6)]#interation
h_values = [1/n for n in n_values]#step size

def f(x, y):
    output = y
    return output

def exactsol(x):
    return np.exp(x)

'''Set zero array for data modification and storage'''
error_array = np.zeros(len(n_values))
time_array = np.zeros(len(n_values))

'''compute error'''
for i, n in enumerate(n_values):#in each interation
    begin_time = time.time()#start time

    '''obtain exact solution'''
    x = np.linspace(0, 1, n+1)
    T_y = np.array([exactsol(x) for x in x])

    '''obtain numerical solution'''
    N_y = np.zeros(n + 1)
    N_y[0] = f(0, 1)
    for j in range(1, n+1):
        N_y[j] = N_y[j-1] + h_values[i] * N_y[j-1]
    end_time = time.time()
    run_time = end_time - begin_time#record time of obtaining numerical solution of each interation
    time_array[i] = run_time#save time of different interation

    '''compute error at each interation in n_values '''
    error = np.max(abs(T_y - N_y))#error under different stepsize (use max error under each step)
    error_array[i] = error#save error of different interation

    '''display result'''
    # print('interation', n, 'time', run_time, 'error', error)
print(error_array)

'''compute convergence rate'''
convergence_rate = np.zeros(len(error_array))

for k in range(1, len(error_array)):
    rate = (math.log(error_array[k])-math.log(error_array[k-1]))/(math.log(h_values[k])-math.log(h_values[k-1]))
    convergence_rate[k] = rate##save convergence rate of each error
    # print('k', k, 'rate', rate)
# print(convergence_rate)
# print(convergence_rate.shape)

'''Figure 2: Convergence Analysis of Forward Euler Method'''
plt.loglog(h_values, error_array, '-o')
plt.xlabel('Step sizes')
plt.ylabel('Error')
plt.title('Convergence Analysis of Forward Euler Method')

plt.show()

'''display results by Table'''
df = pd.DataFrame({'Step size': h_values, 'Error': error_array, 'Convergence rate': convergence_rate, 'Runtime': time_array})



print(df)

























