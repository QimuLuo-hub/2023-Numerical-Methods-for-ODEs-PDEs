# import sympy as sp
#
# def f(x, y):
#     return y * (3 - x ** 2) - 2 * x
#
# def newton_iteration(x0, tol=1e-6, max_iter=100):
#     x = x0
#     for _ in range(max_iter):
#         x_new = x - f(x, y) / f'(x)
#         if abs(x_new - x) < tol:
#             return x_new
#         x = x_new
#     return None
#
# def solve_ode(initial_value, time_steps, dt, target_time):
#     x = initial_value
#     t = 0
#     values = [x]
#     while t < target_time:
#         x_new = x + dt * f(x, y)
#         x = x_new
#         t += dt
#         values.append(x)
#     return values
#
# if __name__ == "__main__":
#     initial_value = 1
#     time_steps = 100
#     dt = 0.01
#     target_time = 10
#
#     sol = solve_ode(initial_value, time_steps, dt, target_time)
#     sp.plot(range(len(sol)), sol)
#     sp.show()
import numpy as np
import matplotlib.pyplot as plt

def backward(t0, tf, h, y0, rhs, exact_y, fprime, max_iterations, epsilon):
    """
    This function implements the backward Euler method to solve initial value problems.
    """
    # Calculate the number of steps
    num_steps = int(np.round((tf - t0) / h))

    # Initialize arrays for storing y and t
    y = np.zeros((num_steps + 1, 1))
    t = np.zeros((num_steps + 1, 1))
    j = np.zeros((num_steps + 1, 1))

    # Set initial value
    y[0] = y0
    t[0] = t0

    # Perform the backward Euler iterations
    for i in range(num_steps):
        t[i + 1] = t[i] + h
        x0 = y[i]
        f = lambda x: -x + y[i] + h * rhs(t[i + 1], x)
        g = lambda x: -1 + h * fprime(t[i + 1], x)

        # Use Newton's method to find the solution
        x1, j[i + 1] = function_newton_test(f, x0, g, max_iterations, epsilon)
        y[i + 1] = x1

    # Calculate the exact solution
    exact_y = exact_y(t)

    # Calculate the error
    error = np.max(np.abs(y - exact_y))

    # # Plot the numerical solution and exact solution
    # plt.figure(2)
    # plt.plot(t, y, '-o', label='Numerical Solution')
    # plt.hold on
    # plt.plot(t, exact_y, '-*', label='Exact Solution')
    # plt.hold off
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.legend()
    # plt.title('Comparison of Numerical and Exact Solutions')
    # plt.show()
    #
    # # Plot the iteration time
    # plt.figure(3)
    # plt.plot(t, j)
    # plt.xlabel('t')
    # plt.ylabel('j')
    # plt.title('Iteration Time')
    # plt.show()

    return y, error

def function_newton_test(f, x0, g, max_iterations, epsilon):
    """
    This function implements the Newton's method to find the solution of a given function.
    """
    for j in range(max_iterations):
        x1 = x0 - f(x0) / g(x0)
        if np.abs(f(x1)) < epsilon:
            break
        x0 = x1

    return x1, j

