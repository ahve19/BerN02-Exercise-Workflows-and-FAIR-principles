# BerN02-Exercise-Workflows-and-FAIR-principles

# Core Computational Snippets: JFNK-Multigrid Burgers' Solver

## 1. System Formulation & Residual Operator

The non-linear residual $F(U) = 0$ for implicit backward Euler temporal discretization is defined as:

$$F(U)_i = U_i - U_{old, i} + \frac{\Delta t}{2 \Delta x} \left( U_i^2 - U_{i-1}^2 \right)$$

Directional derivatives are approximated matrix-free using standard finite differences[cite: 1, 3]:

$$J(U)v \approx \frac{F(U + \varepsilon v) - F(U)}{\varepsilon}, \quad \varepsilon = \frac{10^{-8}}{\Vert{}v\Vert{}_2}$$

```python
# [Code Snippet 1: Residual Vector & Matrix-Free Jacobian-Vector Product]
import numpy as np

# Grid & Parameters
dx = 1 / 4000
nx = int(2 / dx)
x = np.arange(nx) * dx
dt = 0.1

# Initial condition with spatial averaging
x_minus = np.roll(x, 1)
Uold = 0.5 * ((2 + np.sin(np.pi * x_minus)) + (2 + np.sin(np.pi * x)))

def F(U, Uold=Uold, dx=dx, dt=dt):
    """Computes the non-linear residual vector F(U)."""
    Um = np.roll(U, 1)
    return U - Uold + (dt / (2 * dx)) * (U**2 - Um**2)

def Jv(U, v, Uold=Uold, dx=dx, dt=dt):
    """Matrix-free directional derivative J(U) * v."""
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.zeros_like(v)
    eps = 1e-8 / norm_v
    return (F(U + eps * v, Uold, dx, dt) - F(U, Uold, dx, dt)) / eps

# Verify residual evaluation on initial state
res_initial = np.linalg.norm(F(Uold))
print(f"Initial Residual L2 Norm: {res_initial:.6e}")
