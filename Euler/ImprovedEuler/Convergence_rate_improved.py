import time
from Improved_Euler import *
import pandas as pd

'''initialization: h, n, Error_array, time'''
N = 6#number of steps
h = np.zeros(N)
n_values = np.zeros(N)
Error_array = np.zeros(N)
runtime = np.zeros(N)

'''compute error and runtime for each step'''
for i in range(N):
    start_time = time.time()
    '''modify h and n value'''
    h[i] = 10 ** (-(i + 1))#change h value
    n_values[i] = 1 / h[i]

    '''call Improved function to get the numerical y and error(max) for a specific step'''
    y, error = ImprovedEuler(0, 1, h[i], 1)#compute y and error for each step

    '''record time'''
    end_time = time.time()
    runtime[i] = end_time - start_time

    '''record error for each step'''
    Error_array[i] = error

    print('i=', i, 'the step size is', h[i], '\n', 'runtime is', runtime[i])

'''convergence rate'''
print('Error in each step: ', Error_array)
convergence_rates = np.zeros(len(Error_array))
convergence_rates[1:] = np.diff(np.log(Error_array)) / np.diff(np.log(h))

'''display results by Table and graph'''

'''log the x and y coordinates same for easy observation'''
plt.loglog(n_values, Error_array, '-o')
plt.xlabel('interation n_values(1/h)')
plt.ylabel('Error')
plt.title('Convergence Analysis of Improved Euler Method')


''' plot and save the image'''
fig = plt.gcf()
fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
filename = 'Convergence_Improved_Euler_Method.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight')
plt.show()
plt.close('all')

'''Table'''
df = pd.DataFrame({'Step size': h, 'Error': Error_array, 'Convergence rate': convergence_rates, 'Runtime': runtime})
'''convert data type'''
df['Step size'] = df['Step size'].apply(lambda x: f"{x:.2e}")#Convert to scientific notation with anonymous functions （Keep one decimal place）
df['Error'] = df['Error'].apply(lambda x: f"{x:.2e}")
df['Convergence rate'] = df['Convergence rate'].apply(lambda x: f"{x:.2f}")#（Keep 2 decimal place）
print(df)





