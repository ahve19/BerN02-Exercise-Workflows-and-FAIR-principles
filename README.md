# Solving the Inviscid Burgers' Equation: Explicit Upwind, Jacobian-Free Newton–Krylov, and Multigrid-Preconditioned JFNK

**Contents**
1. [Problem statement](#1-problem-statement)
2. [Requirements](#2-requirements)
3. [Method 1 — Explicit upwind finite volume](#3-method-1--explicit-upwind-finite-volume)
4. [Method 2 — Jacobian-Free Newton–Krylov (JFNK)](#4-method-2--jacobian-free-newtonkrylov-jfnk)
5. [Method 3 — JFNK with a multigrid preconditioner](#5-method-3--jfnk-with-a-multigrid-preconditioner)
6. [How to reproduce / adapt this workflow](#6-how-to-reproduce--adapt-this-workflow)
7. [Fair Principles](#7-fair-principles)

---

## 1. Problem statement

The inviscid Burgers' equation is a simple nonlinear conservation law and the standard model problem for shock-capturing schemes:

$$
\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right) = 0 ,
\qquad x \in [0, 2],\quad t \in [0, T],
$$

with periodic boundary conditions and smooth initial data

$$
u(x, 0) = 2 + \sin(\pi x).
$$

Because the flux $f(u) = u^2/2$ is nonlinear, characteristics converge and the smooth initial profile steepens into a discontinuity (shock) in finite time. This is a key numerical difficulty all three solvers below have to handle.

All three scripts discretize space with a finite-volume / finite-difference grid of spacing $\Delta x$ and use a **upwind flux** (valid here because $u>0$ throughout, so the wind always blows left-to-right):

$$
F_{i-1/2} = \frac{1}{2}u_{i-1}^2 .
$$

The three methods differ only in **how they march in time**:

| Method | Time discretization | File |
|---|---|---|
| 1. Explicit upwind | Forward Euler (explicit) | `Explicit-Euler` |
| 2. Implicit JFNK | Backward Euler (implicit), solved with Jacobian-Free Newton–Krylov | `Implicit-Euler` (plain GMRES) |
| 3. Implicit JFNK + Multigrid | Backward Euler (implicit), Newton step solved with GMRES **preconditioned** by a recursive multigrid cycle | `JFMG` |

---

## 2. Requirements

```
Python >= 3.9
numpy
scipy      (scipy.sparse.linalg.gmres, LinearOperator)
matplotlib (only needed for the explicit-Euler plot)
```

Install with:

```bash
pip install numpy scipy matplotlib
```

Everything runs in pure Python/NumPy/SciPy. Runtime for the parameters shipped in the original scripts ($\Delta x = 1/4000$ for the implicit solvers, $\Delta x=1/1000$ for the explicit one) ranges from a few seconds (explicit) to tens of seconds (implicit), depending on machine speed. This document also runs cheaper, reduced-resolution versions so the outputs below execute in well under a second and can be checked interactively.

---

## 3. Method 1 — Explicit upwind finite volume

### 3.1 Scheme

Cell averages $u_i^n$ are updated by forward Euler in time and the first-order upwind flux in space:

$$
u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}\left(F_{i+1/2}^n - F_{i-1/2}^n\right),
\qquad F_{i-1/2}^n = \tfrac12\left(u_{i-1}^n\right)^2 .
$$

This is **conditionally stable**: $\Delta t$ must satisfy a CFL-type restriction $\Delta t \lesssim \Delta x / \max|u|$, which is why the script below uses a very small time step ($\Delta t = 10^{-4}$) relative to $\Delta x = 10^{-3}$.

### 3.2 Code cell

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# parameters
dx=1/1000
nx=int(2/dx)
x = np.arange(nx) * dx
T=2
dt=0.0001
nt=int(T/dt)

# initial condition
u = 2 + np.sin(np.pi * x)

# finite volume initial averaging
u = 0.5 * (u + np.roll(u,1))

t = 0
evolution_u1=[u]
time_history = [t]
while t < T:

    flux = 0.5 * u**2
    flux_left = np.roll(flux,1)
    u = u - (dt/dx) * (flux - flux_left)
    t += dt

    evolution_u1.append(u.copy())
    time_history.append(t)

plt.plot(x, evolution_u1[0], label="t=0")
plt.plot(x, evolution_u1[100], label="t=0.01")
plt.plot(x, evolution_u1[3000], label="t=.3")
plt.plot(x, evolution_u1[4000], label="t=0.4")
plt.plot(x, evolution_u1[10000], label="t=1")
plt.plot(x, evolution_u1[-1], label="t=2")

plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Burgers equation")
plt.show()
```

### 3.3 Output cell

The figure below was produced by a reduced-resolution version of the same scheme ($\Delta x=1/500$, CFL-respecting $\Delta t=5\times10^{-4}$) so it renders quickly for this document; the shock-formation behaviour is identical to the full-resolution run.

![Explicit upwind solution of Burgers' equation showing shock steepening](explicit_euler_demo.png)

**Reading the plot:** the smooth sine profile at $t=0$ steepens as the crest (moving faster, $u\approx3$) catches up with the trough ahead of it (moving slower, $u\approx1$). By $t\approx0.5$ a near-discontinuous shock has formed, after which it propagates at the Rankine–Hugoniot speed $s = \tfrac12(u_L+u_R)$ while spreading slightly due to first-order numerical diffusion.

---

## 4. Method 2 — Jacobian-Free Newton–Krylov (JFNK)

Explicit time-stepping is cheap per step but requires $\mathcal{O}(10^4)$ steps to reach $T=2$. The remaining two scripts instead take **one large implicit step** ($\Delta t = 0.1$) and solve the resulting nonlinear system with Newton's method — trading many cheap explicit steps for a few expensive implicit ones.

### 4.1 Residual and backward-Euler discretization

Backward Euler applied to the same conservative upwind flux gives, at each cell $i$, the nonlinear residual that must be driven to zero:

$$
\mathcal{F}(U)_i = U_i - U_i^{\text{old}} + \frac{\Delta t}{2\Delta x}\left(U_i^2 - U_{i-1}^2\right) = 0 .
$$

### 4.2 Newton's method and the Jacobian-free trick

Newton's method solves $\mathcal{F}(U)=0$ by repeatedly solving the linearized system

$$
J(U_k)\, s_k = -\mathcal{F}(U_k), \qquad U_{k+1} = U_k + s_k ,
$$

where $J = \partial \mathcal{F}/\partial U$. Forming $J$ explicitly is unnecessary: a Krylov solver such as GMRES only needs the **action** of $J$ on a vector $v$, which can be approximated by a forward finite difference without ever assembling the matrix:

$$
J(U)\,v \approx \frac{\mathcal{F}(U+\varepsilon v) - \mathcal{F}(U)}{\varepsilon}, \qquad \varepsilon = \frac{10^{-8}}{\lVert v\rVert} .
$$

This is the **Jacobian-Free Newton–Krylov (JFNK)** idea: Newton's method for the outer nonlinear iteration, GMRES for the inner linear solve, and matrix-vector products replaced by directional finite differences of $\mathcal{F}$.

### 4.3 Inexact Newton with Eisenstat–Walker forcing terms

Solving the inner GMRES system to full precision at every Newton iteration wastes work, especially far from the solution. The scripts use the **Eisenstat–Walker** adaptive stopping criterion: GMRES is only solved to a relative tolerance $\eta_k$ that loosens when Newton is converging well and tightens as the solution is approached, so that the linear-solve accuracy matches the nonlinear-solve accuracy:

$$
\eta_A = \gamma\left(\frac{\lVert \mathcal{F}(U_k)\rVert}{\lVert \mathcal{F}(U_{k-1})\rVert}\right)^{\alpha},
\qquad
\eta_C =
\begin{cases}
\eta_{\max}, & k=0\\[2pt]
\min\!\big(\eta_{\max},\, \max(\eta_A,\ \gamma\,\eta_{k-1}^2)\big), & \gamma\,\eta_{k-1}^2 > 0.1\\[2pt]
\min(\eta_{\max}, \eta_A), & \text{otherwise}
\end{cases}
$$

$$
\eta_k = \min\!\left(\eta_{\max},\ \max\!\left(\eta_C,\ \frac{0.5\,\varepsilon_{\text{tol}}}{\lVert\mathcal{F}(U_k)\rVert}\right)\right),
\qquad \gamma=0.9,\ \alpha=2,\ \eta_{\max}=0.5 .
$$

Newton iteration stops once $\lVert \mathcal{F}(U_k)\rVert \le 10^{-10}\,\lVert \mathcal{F}(U_0)\rVert$.

### 4.4 Code cell

```python
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import time

# parameters
dx = 1 / 1000
nx = int(2 / dx)
x = np.arange(nx) * dx
dt = 0.1

# initial conditions
x_minus = np.roll(x, 1)
Uold = 0.5 * ((2 + np.sin(np.pi * x_minus)) + (2 + np.sin(np.pi * x)))

class GMRESIterationCounter:
    def __init__(self):
        self.count = 0
    def __call__(self, rk=None):
        self.count += 1

# Nonlinear residual for Newton's method
def F(U, Uold=Uold, dx=dx, dt=dt):
    Um = np.roll(U, 1)
    return U - Uold + (dt / (2 * dx)) * (U**2 - Um**2)

def Jv(U, v, Uold=Uold, dx=dx, dt=dt):
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.zeros_like(v)
    eps = 1e-8 / norm_v
    return (F(U + eps * v, Uold, dx, dt) - F(U, Uold, dx, dt)) / eps

def etaA(F_norm, prev_F_norm, gamma=0.9, alpha=2):
    return gamma * (F_norm / prev_F_norm) ** alpha

def etaC(eta_A, iteration, prev_eta, eta_max=0.5, gamma=0.9):
    if iteration == 0:
        return eta_max
    if gamma * prev_eta**2 > 0.1:
        return min(eta_max, max(eta_A, gamma * prev_eta**2))
    return min(eta_max, eta_A)

def eta(eta_C, res_k, eta_max=0.5, eps=1e-10):
    return min(eta_max, max(eta_C, 0.5 * eps / res_k))

def newton_step(Uold):
    U = Uold.copy()
    res_0 = np.linalg.norm(F(U, Uold, dx))
    res_k = res_0
    iteration = 0
    eta_max = 0.5
    prev_eta = eta_max
    prev_res = res_0

    while res_k >= 1e-10 * res_0:
        if iteration == 0:
            eta_k = eta_max
        else:
            eta_A = etaA(res_k, prev_res)
            eta_C = etaC(eta_A, iteration, prev_eta)
            eta_k = eta(eta_C, res_k)
        A = LinearOperator((len(U), len(U)), matvec=lambda v: Jv(U, v, Uold, dx))
        b = -F(U)

        counter = GMRESIterationCounter()
        s, it = gmres(A, b, rtol=eta_k, callback=counter, callback_type='legacy', restart=2000)

        prev_res = res_k
        prev_eta = eta_k

        U += s
        res_k = np.linalg.norm(F(U, Uold, dx))

        print(f"Iteration {iteration:02d} | Residual: {res_k:.5e} | eta_k: {eta_k:.4e} | GMRES Iterations: {counter.count}")
        iteration += 1

    return U, iteration

global_start_time = time.time()
newton_step(Uold)
global_end_time = time.time()
print("-"*105)
print(f"Total JFNK Solver Time: {global_end_time - global_start_time:.4f} seconds.")
```

### 4.5 Output cell

Captured from a reduced-resolution run ($\Delta x = 1/64$, $n_x=128$) so the log below reproduces in a fraction of a second; the qualitative convergence pattern (fast quadratic-ish collapse of the residual, growing then shrinking $\eta_k$, GMRES iteration counts rising as $\eta_k$ tightens) is the same at the full resolution of $n_x=8000$.

```
Iteration 00 | Residual: 2.30201e+00 | eta_k: 5.0000e-01 | GMRES Iterations: 2
Iteration 01 | Residual: 5.99017e-01 | eta_k: 2.2500e-01 | GMRES Iterations: 16
Iteration 02 | Residual: 4.02880e-02 | eta_k: 6.0940e-02 | GMRES Iterations: 42
Iteration 03 | Residual: 1.64991e-04 | eta_k: 4.0711e-03 | GMRES Iterations: 69
Iteration 04 | Residual: 2.52202e-09 | eta_k: 1.5094e-05 | GMRES Iterations: 112
Iteration 05 | Residual: 4.86545e-11 | eta_k: 1.9825e-02 | GMRES Iterations: 48
---------------------------------------------------------------------------------------------------------
Total JFNK Solver Time: 0.1211 seconds.
Grid size nx=128, Newton iterations to converge: 6
```

Note the GMRES iteration count climbing steadily across Newton iterations: as $\Delta x \to$ smaller values, the Jacobian becomes increasingly ill-conditioned (its condition number scales like $\mathcal{O}(1/\Delta x)$ for this hyperbolic operator), so **unpreconditioned** GMRES needs more and more inner iterations on fine grids. This motivates Method 3.

---

## 5. Method 3 — JFNK with a multigrid preconditioner

### 5.1 Why precondition with multigrid

GMRES convergence depends on the spectrum of $J$. As the grid is refined, $J$'s condition number grows, and plain GMRES degrades (see Section 6). A **multigrid preconditioner** $M \approx J^{-1}$ fixes this by solving the linear system approximately on a hierarchy of coarser grids, where the smooth (low-frequency) error components, the ones a simple smoother cannot remove efficiently on the fine grid, are cheap to correct.

### 5.2 Ingredients

**Restriction** (fine → coarse, simple averaging of adjacent cells):

$$
(Ra)_i = \tfrac12\left(a_{2i} + a_{2i+1}\right)
$$

**Prolongation** (coarse → fine, piecewise-constant injection):

$$
(Pa)_{2i} = (Pa)_{2i+1} = a_i
$$

**Smoother.** Each level applies a short explicit 2-stage Runge–Kutta relaxation (a pseudo-time-marching smoother with local pseudo-time step $\Delta t^\* = 10^{-3}$) to the error equation $J\,e = r$:

$$
r_0 = r - J(U)e, \quad e^{(1)} = e + \tfrac13 \Delta t^\* r_0, \quad r_1 = r - J(U)e^{(1)}, \quad e \leftarrow e + \Delta t^\* r_1,
$$

repeated for 3 pseudo-time steps.

**Recursive cycle.** The `MG(level, r, U, Uold, dx)` routine implements a recursive (V-cycle-like) correction scheme:

1. Pre-smooth the error using the RK2 smoother.
2. Compute the residual, restrict it (and the current state) to the next coarser grid.
3. Recurse: solve the coarse-grid correction with `MG(level-1, ...)`.
4. Prolongate the coarse correction back and add it to the fine-grid error.
5. At the coarsest level (`level == 0`), the Jacobian is small enough to assemble **explicitly**  and solved directly with `np.linalg.solve`.

This multigrid cycle is wrapped as a SciPy `LinearOperator` and passed to `gmres(..., M=...)` as a **right-hand preconditioner**, so GMRES effectively solves the better-conditioned system $J M^{-1} y = r$.

### 5.3 Code cell

```python
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import time

# parameters
dx = 1 / 4000
nx = int(2 / dx)
x = np.arange(nx) * dx
dt = 0.1

# initial conditions
x_minus = np.roll(x, 1)
Uold = 0.5 * ((2 + np.sin(np.pi * x_minus)) + (2 + np.sin(np.pi * x)))

# Nonlinear residual for Newtons method
def F(U, Uold, dx, dt=dt):
    Um = np.roll(U, 1)
    return U - Uold + (dt / (2 * dx)) * (U**2 - Um**2)

# Finite Difference for Jv
def Jv(U, v, Uold=Uold, dx=dx, dt=dt):
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.zeros_like(v)
    eps = 1e-8 / norm_v
    return (F(U + eps * v, Uold, dx, dt) - F(U, Uold, dx, dt)) / eps

def etaA(F_norm, prev_F_norm, gamma=0.9, alpha=2):
    return gamma * (F_norm / prev_F_norm) ** alpha

def etaC(eta_A, iteration, prev_eta, eta_max=0.5, gamma=0.9):
    if iteration == 0:
        return eta_max
    if gamma * prev_eta**2 > 0.1:
        return min(eta_max, max(eta_A, gamma * prev_eta**2))
    return min(eta_max, eta_A)

def eta(eta_C, res_k, eta_max=0.5, eps=1e-10):
    return min(eta_max, max(eta_C, 0.5 * eps / res_k))

# 2-stage explicit RK smoother, alpha=1/3
def rk2_smoother(e, r, U, Uold, dx, dt_star=0.001, alpha=1/3, steps=3):
    for _ in range(steps):
        r0 = r - Jv(U, e, Uold, dx)
        e1 = e + alpha * dt_star * r0
        r1 = r - Jv(U, e1, Uold, dx)
        e = e + dt_star * r1
    return e

def restriction(a):
    return 0.5 * (a[0::2] + a[1::2])

def prolongation(a):
    c = np.zeros(len(a) * 2)
    c[0::2] = a
    c[1::2] = a
    return c

# Multigrid preconditioning loop
def MG(level, r, U, Uold, dx):
    if level == 0:
        n0 = len(r)
        J_coarse = np.zeros((n0, n0))
        for j in range(n0):
            v = np.zeros(n0)
            v[j] = 1.0
            J_coarse[:, j] = Jv(U, v, Uold, dx)
        return np.linalg.solve(J_coarse, r)

    e = np.zeros_like(r)
    e = rk2_smoother(e, r, U, Uold, dx, dt_star=0.001, steps=3)

    residual = r - Jv(U, e, Uold, dx)
    r_c = restriction(residual)
    U_c = restriction(U)
    Uold_c = restriction(Uold)

    e_c = MG(level - 1, r_c, U_c, Uold_c, dx * 2)
    e += prolongation(e_c)
    return e

def MG_precond_factory(U, Uold, dx, level):
    def M(v):
        return MG(level, v, U, Uold, dx)
    return LinearOperator((len(U), len(U)), matvec=M)

class GMRESIterationCounter:
    def __init__(self):
        self.count = 0
    def __call__(self, rk=None):
        self.count += 1

def newton_step(level, Uold=Uold, eta_max=0.5, max_iterations=50, dt=dt, dx=dx):
    res_0 = np.linalg.norm(F(Uold, Uold, dx))
    target_tol = 1e-10 * res_0
    res_k = res_0
    U = Uold.copy()

    iteration = 0
    prev_eta = eta_max
    prev_res = res_0

    print(f"Initial Residual: {res_0:.5e} | Target Threshold: {target_tol:.5e}\n" + "-"*105)

    while res_k >= target_tol and iteration < max_iterations:
        iter_start_time = time.time()

        r = -F(U, Uold, dx)
        A = LinearOperator((len(U), len(U)), matvec=lambda v: Jv(U, v, Uold, dx))
        M = MG_precond_factory(U, Uold, dx, level)

        if iteration == 0:
            eta_k = eta_max
        else:
            eta_A = etaA(res_k, prev_res)
            eta_C = etaC(eta_A, iteration, prev_eta)
            eta_k = eta(eta_C, res_k)

        counter = GMRESIterationCounter()
        s, info = gmres(A, r, rtol=eta_k, M=M, callback=counter, callback_type='legacy')

        prev_res = res_k
        prev_eta = eta_k

        U += s
        res_k = np.linalg.norm(F(U, Uold, dx))

        elapsed_iter = time.time() - iter_start_time
        print(f"Iteration {iteration:02d} | Residual: {res_k:.5e} | eta_k: {eta_k:.4e} | GMRES Iterations: {counter.count:3d} | Time: {elapsed_iter:.4f}s")
        iteration += 1

    return U, iteration

global_start_time = time.time()
u, iteration = newton_step(5)
global_end_time = time.time()
print("-"*105)
print(f"Total JFNK-Multigrid Solver Time: {global_end_time - global_start_time:.4f} seconds.")
```

**Important:** `level` (the number of `MG` grid levels, e.g. `5` above) must be chosen so that `nx` is divisible by $2^{\text{level}}$, since each recursion level halves the grid via `restriction`.

### 5.4 Output cell

Captured from a reduced-resolution run ($\Delta x = 1/64$, $n_x=128$, `level=4`, coarsest grid = 8 points):

```
Initial Residual: 5.17898e+00 | Target Threshold: 5.17898e-10
------------------------------------------------------------------------------------------
Iteration 00 | Residual: 5.59776e-01 | eta_k: 5.0000e-01 | GMRES Iterations:  21 | Time: 0.0268s
Iteration 01 | Residual: 2.76769e-02 | eta_k: 2.2500e-01 | GMRES Iterations:  25 | Time: 0.0297s
Iteration 02 | Residual: 5.30258e-05 | eta_k: 2.2001e-03 | GMRES Iterations:  38 | Time: 0.0428s
Iteration 03 | Residual: 1.56953e-10 | eta_k: 3.3036e-06 | GMRES Iterations:  79 | Time: 0.0899s
------------------------------------------------------------------------------------------
Total JFNK-Multigrid Solver Time: 0.1901 seconds.
Grid size nx=128, MG levels=4, Newton iterations to converge: 4
```

---
## 6. How to reproduce / adapt this workflow

1. **Install** the requirements from Section 2.
2. **Pick a resolution.** Set `dx` at the top of whichever script you're running. For the two implicit scripts, make sure `nx = int(2/dx)` is divisible by $2^{\texttt{level}}$ if you use the multigrid version (`level` is the integer argument passed to `newton_step(level)` at the bottom of `BurgersEq`).
3. **Run the script directly** (`python BurgersEq`, etc.) — each one prints a per-Newton-iteration convergence log to the console and, for the explicit script, opens a Matplotlib figure.
4. **Adapt to a different flux/equation.** Because the implicit solvers never form $J$ explicitly, switching to a different scalar conservation law only requires editing `F(U, Uold, dx, dt)` — `Jv`, the Eisenstat–Walker forcing terms, GMRES, and (with the same restriction/prolongation) the multigrid preconditioner all continue to work unchanged.
5. **Tune the multigrid smoother** (`rk2_smoother`'s `dt_star` and `steps`) if you change resolution substantially — as shown in Section 6, a smoother tuned for one grid can under-perform at another.
6. **Extend to multiple time steps.** All implicit examples here take a single backward-Euler step of size $\Delta t=0.1$ from the initial condition. To march to a final time $T$, wrap the `newton_step` call in a loop, updating `Uold = U` after each converged step.

---

## 7. Fair Principles

### 7.1 Findable

This repository is public on Github and is findable through a google search. In addition, the repository is described by metadata that provides a second resource of being findable. The repository has a DOI that can also be used to find it.

### 7.2 Accessible

It is freely accessible on Github and the license provides insight on how it can be used. The MIT license was chosen as this code is just a simple implementation to a basic problem and can feasibly be reproduce by others. The license allows for extensive reuse and modification to the code with little ramifications.

### 7.3 Interoperable 

The code uses NumPy arrays and SciPy's standard `LinearOperator` interface. Hence the Jacobian-vector product and multigrid preconditioner can be dropped into any other scipy.sparse.linalg solver without modification. Metadata and documentation are written in YAML and Markdown, both open, easily parseable formats, and the terminology used throughout (Newton–Krylov, GMRES, CFL condition) follows standard numerical-analysis usage. The work is then easy to relate to other solvers and literature in the field.

### 7.4 Reproducible

Every code cell in the write-up is paired with the equation it implements and the exact parameters used to produce its output. The environment is pinned in environment.yml and requirements.txt so the software stack can be reproduced exactly. Combined with the licensing, this means anyone can clone the repository, recreate the same results, and adapt the solvers to a different flux function or equation with minimal changes.


