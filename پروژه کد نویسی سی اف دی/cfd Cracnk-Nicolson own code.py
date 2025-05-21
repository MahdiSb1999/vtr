import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Thomas algorithm function (as defined above)
def thomas_algorithm(a, b, c, d):
    n = len(b)
    b = b.copy()
    d = d.copy()
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


# Diffusion equation solver
def solve_diffusion(L, T, N, M, nu, f, a_func, b_func):
    # Discretization parameters
    delta_y = L / N
    delta_t = T / M
    alpha = nu * delta_t / (2 * delta_y**2)

    # Spatial grid
    y = np.linspace(0, L, N + 1)

    # Initial condition
    u = f(y)

    # Solution array
    u_all = np.zeros((M + 1, N + 1))
    u_all[0, :] = u

    # Define tridiagonal matrix diagonals (length N-1)
    a = -alpha * np.ones(N - 1)
    a[0] = 0  # No subdiagonal for first row
    b = (1 + 2 * alpha) * np.ones(N - 1)
    c = -alpha * np.ones(N - 1)
    c[-1] = 0  # No superdiagonal for last row

    # Time-stepping loop
    for n in range(M):
        t_next = (n + 1) * delta_t
        u_new = np.zeros(N + 1)

        # Boundary conditions
        u_new[0] = a_func(t_next)
        u_new[-1] = b_func(t_next)

        # Right-hand side for interior points
        R = alpha * u[:-2] + (1 - 2 * alpha) * u[1:-1] + alpha * u[2:]
        R[0] += alpha * u_new[0]  # Boundary contribution
        R[-1] += alpha * u_new[-1]  # Boundary contribution

        # Solve using Thomas algorithm
        u_new[1:-1] = thomas_algorithm(a, b, c, R)

        # Update solution
        u = u_new.copy()
        u_all[n + 1, :] = u

    return y, u_all


# Problem parameters
L = 0.04  # Domain length
T = 1.08  # Total time
N = 40  # Spatial steps
M = 540  # Time steps
nu = 0.000217  #
dy = L / N
dt = T / M


# Initial and boundary conditions
def f(y):
    return np.zeros_like(y)  # u(y, 0) = 0


def a_func(t):
    return 0  # u(0, t) = 0


def b_func(t):
    return 40  # u(L, t) = 40


# Solve and plot
y, u_all = solve_diffusion(L, T, N, M, nu, f, a_func, b_func)
y_coords = np.arange(0, L + dy, dy)
t_coords = np.arange(0, T + dy, dt)


df = pd.DataFrame(u_all.T, index=y_coords, columns=t_coords)
df.to_csv("outputCrank-Nicolson.csv", float_format="%.3f")
