import time
from Five_point_v1 import *
import pandas as pd

'''initialization: h, n, Error_array, time'''

N = 3 #number of values for h
h = np.zeros(N)
n_values = np.zeros(N)
Error_array = np.zeros(N)
runtime = np.zeros(N)

'''compute error and runtime for each step'''
for i in range(N):
    start_time = time.time()
    '''modify h and n value'''
    h[i] = 10 ** (-(i + 1))#change h value
    n_values[i] = round(1/h[i])

    '''call function to get the error(max) for each value of h'''
    error = solve_poisson_equation(0, 1, h[i])

    end_time = time.time()
    runtime[i] = end_time - start_time

    '''record error for each step'''
    Error_array[i] = error
    print('the step size is', h[i], '\n', 'runtime is', runtime[i])


#compute convergence rate
convergence_rates = np.zeros(len(Error_array))
convergence_rates[1:] = np.diff(np.log(Error_array)) / np.diff(np.log(h))
plt.loglog(n_values, Error_array, '-o')
plt.xlabel('interation n_values(1/h)')
plt.ylabel('Error')
plt.title('Convergence Analysis of Five-Points Difference Method')

fig = plt.gcf()
fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
filename = ('Convergence_Five_Points_Difference_Method.pdf')
plt.savefig(filename, format='pdf', bbox_inches='tight')
plt.show()

#Table
df = pd.DataFrame({'Step size': h, 'Error': Error_array, 'Convergence rate': convergence_rates, 'Runtime': runtime})
df['Step size'] = df['Step size'].apply(lambda x: f"{x:.1e}")
df['Error'] = df['Error'].apply(lambda x: f"{x:.2e}")
df['Convergence rate'] = df['Convergence rate'].apply(lambda x: f"{x:.2f}")
print(df)





