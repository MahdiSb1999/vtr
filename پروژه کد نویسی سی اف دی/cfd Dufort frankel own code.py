import numpy as np
import pandas as pd

nu = 0.000217
delta_t = 0.003
u_0 = 40
h = 0.04
t_final = 1.08
delta_y = 0.001
Ny = int(h / (delta_y) + 1)
Nt = int(t_final / delta_t) + 1

u = np.zeros((Ny, Nt))
u[0, :] = 0
u[1:-1, 0] = 0
u[-1, :] = u_0
d = (nu * delta_t) / (delta_y**2)
u_old = u.copy()

for j in range(1, Ny - 1):
    u[j, 1] = u_old[j, 0] + d * (u_old[j + 1, 0] + -2 * u_old[j, 0] + u_old[j - 1, 0])

u_old = u.copy()

for i in range(2, Nt):
    for j in range(1, Ny - 1):
        u[j, i] = (1 - 2 * d) / (1 + 2 * d) * u_old[j, i - 2] + (2 * d) / (
            1 + 2 * d
        ) * (u_old[j + 1, i - 1] + u_old[j - 1, i - 1])
    u_old = u.copy()
y_coords = np.arange(0, h + delta_y, delta_y)[:Ny]
t_coords = np.arange(0, t_final + delta_t, delta_t)[:Nt]
df = pd.DataFrame(u, index=y_coords, columns=t_coords)
df.to_csv("outputdufort frankelcase2.csv", float_format="%.3f")
