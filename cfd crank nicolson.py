import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

def solve_diffusion(L, T, N, M, nu, f, a, b):
    """
    Solve the diffusion equation ∂u/∂t = ν ∂²u/∂y² using the Crank-Nicolson method.
    
    Parameters:
    -----------
    L : float
        Length of the spatial domain [0, L].
    T : float
        Total time of the simulation [0, T].
    N : int
        Number of spatial grid points (excluding boundaries).
    M : int
        Number of time steps.
    nu : float
        Diffusivity constant (ν).
    f : callable
        Initial condition function, u(y, 0) = f(y).
    a : callable
        Boundary condition at y = 0, u(0, t) = a(t).
    b : callable
        Boundary condition at y = L, u(L, t) = b(t).
    
    Returns:
    --------
    y : ndarray
        Array of spatial grid points.
    u_all : ndarray
        2D array containing the solution u(y, t) at all time steps.
    """
    # Discretization parameters
    delta_y = L / N
    delta_t = T / M
    alpha = nu * delta_t / (2 * delta_y**2)
    
    # Spatial grid
    y = np.linspace(0, L, N + 1)
    
    # Initial condition
    u = f(y)
    
    # Array to store solution at all time steps
    u_all = np.zeros((M + 1, N + 1))
    u_all[0, :] = u
    
    # Set up tridiagonal matrix A (for interior points i=1 to N-1)
    lower = -alpha * np.ones(N - 2)          # Subdiagonal
    main = (1 + 2 * alpha) * np.ones(N - 1)  # Main diagonal
    upper = -alpha * np.ones(N - 2)          # Superdiagonal
    
    # Format for solve_banded: ab has shape (3, N-1)
    ab = np.zeros((3, N - 1))
    ab[0, 1:] = upper  # Upper diagonal, shifted
    ab[1, :] = main    # Main diagonal
    ab[2, :-1] = lower # Lower diagonal, shifted
    
    # Time-stepping loop
    for n in range(M):
        t_next = (n + 1) * delta_t
        u_new = np.zeros(N + 1)
        
        # Apply boundary conditions
        u_new[0] = a(t_next)
        u_new[N] = b(t_next)
        
        # Compute right-hand side R for interior points
        R = (alpha * u[:-2] + 
             (1 - 2 * alpha) * u[1:-1] + 
             alpha * u[2:])
        R[0] += alpha * u_new[0]   # Boundary contribution at y=0
        R[-1] += alpha * u_new[N]  # Boundary contribution at y=L
        
        # Solve the tridiagonal system A u^{n+1} = R
        u_new[1:-1] = solve_banded((1, 1), ab, R)
        
        # Update solution
        u = u_new.copy()
        u_all[n + 1, :] = u_new
    
    return y, u_all

# Define problem parameters
L = 1.0      # Spatial domain length
T = 1.0      # Total time
N = 100      # Number of spatial steps
M = 1000     # Number of time steps
nu = 0.01    # Diffusivity

# Initial and boundary condition functions
def f(y):
    """Initial condition: u(y, 0) = sin(π y / L)"""
    return np.sin(np.pi * y / L)

def a(t):
    """Boundary condition at y = 0"""
    return 0

def b(t):
    """Boundary condition at y = L"""
    return 0

# Solve the diffusion equation
y, u_all = solve_diffusion(L, T, N, M, nu, f, a, b)

# Plot the solution at the final time
plt.plot(y, u_all[M, :], label=f't = {T}')
plt.xlabel('y')
plt.ylabel('u(y, t)')
plt.title('Diffusion Equation Solution (Crank-Nicolson)')
plt.legend()
plt.grid(True)
plt.show()