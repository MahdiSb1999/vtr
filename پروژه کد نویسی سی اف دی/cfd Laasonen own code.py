import numpy as np
import pandas as pd


def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d.
    a: diagonal elements (length n)
    b: sub-diagonal elements (length n-1)
    c: super-diagonal elements (length n-1)
    d: right-hand side (length n)
    Returns: solution x
    """
    n = len(d)
    H = np.zeros(n - 1)
    G = np.zeros(n)
    x = np.zeros(n)

    # Forward elimination
    H[0] = c[0] / a[0]
    G[0] = d[0] / a[0]

    for i in range(1, n - 1):
        denom = a[i] - b[i - 1] * H[i - 1]
        H[i] = c[i] / denom
        G[i] = (d[i] - b[i - 1] * G[i - 1]) / denom

    G[n - 1] = (d[n - 1] - b[n - 2] * G[n - 2]) / (a[n - 1] - b[n - 2] * H[n - 2])

    # Back substitution
    x[n - 1] = G[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = G[i] - H[i] * x[i + 1]

    return x


# Example setup
U_0 = 40
nu = 0.000217  # viscosity
dy = 0.001  # spatial step
dt = 0.01  # time step
H = 0.04
t_final = 1.08
Ny = int(H/dy+1)  # number of nodes
Nt = int(t_final/dt + 1)  # number of time_steps

# Tridiagonal matrix coefficients
alpha = nu * dt / (dy**2)
c = -alpha  # off-diagonal
a = 1 + 2 * alpha  # diagonal
b = -alpha  # off-diagonal (same as c for symmetry)

# Initialize arrays
a_diag = np.array([a] * (Ny))  # diagonal
b_sub = np.array([b] * (Ny - 1))  # sub-diagonal
c_sup = np.array([c] * (Ny - 1))  # super-diagonal

a_diag[0] = 1
a_diag[-1] = 1
b_sub[-1] = 0
c_sup[0] = 0
u = np.zeros(Ny)
u[0:-1] = 0
u[-1] = U_0




# Store solutions for all time steps
u_history = [u.copy()]

# Time-stepping loop
for step in range(1,Nt):
    # Update RHS
    d = u.copy()  # u^n
    d[0] = 0  # BC: u_1 = 0
    d[-1] = U_0  # BC: u_ny = 40

    # Solve for u^{n+1}
    u = thomas_algorithm(a_diag, b_sub, c_sup, d)

    # Store solution
    u_history.append(u.copy())

# Convert history to array
u_history = np.array(u_history)
u_final = u_history.T

y_coords = np.arange(0, H + dy, dy)[:Ny]
t_coords = np.arange(0, t_final + dt, dt)[:Nt]
df = pd.DataFrame(u_final, index=y_coords, columns=t_coords)
df.to_csv("outputlaasonen.csv", float_format="%.3f")

