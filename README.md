# Numerical Methods for ODEs and PDEs

From-scratch Python implementations of classical numerical methods for ordinary and partial differential equations. Each method comes with a convergence study against an analytical solution: the maximum error and runtime are recorded for a range of step sizes, and the observed order of convergence is computed as

$$p_i = \frac{\log E_{i} - \log E_{i-1}}{\log h_{i} - \log h_{i-1}}.$$

Plots of the numerical and exact solutions, and log–log convergence plots, are saved as PDFs next to the scripts.

## Methods

### ODE solvers — `Euler/`, `Rungekuta/`

Test problem: $y' = y$, $y(0) = 1$ on $[0, 1]$, with exact solution $y = e^x$. The step size runs over $h = 10^{-1}, \dots, 10^{-6}$.

| Folder / script | Method | Notes |
|---|---|---|
| `Euler/ForwardEuler/` | Forward (explicit) Euler | `Forward_Euler_v2.py`, convergence scripts |
| `Euler/BackwardEuler/` | Backward (implicit) Euler | Each implicit step is solved with Newton's method; test problem $y' = \sin(t)\,y$, $y = e^{-\cos t}$ |
| `Euler/ImprovedEuler/` | Improved Euler (Heun) | Euler predictor + trapezoidal corrector |
| `Rungekuta/runge2order.py` | General 2-stage, 2nd-order Runge–Kutta | Parametrised by the weight $m_2$; the convergence study uses $m_2 = 3/4$ (Ralston) |
| `Rungekuta/runge3order.py` | General 3-stage, 3rd-order Runge–Kutta | Parametrised by the nodes $p$, $l$; the convergence study uses $p = 1/2$, $l = 1$ (Kutta's method) |
| `Rungekuta/runge4order.py` | 4-stage Runge–Kutta | |
| `Rungekuta/*order_convergence.py` | Convergence tables and log–log plots | |

### Finite differences — `FDM/`

The **five-point stencil** for the 2D Poisson equation on $[0,1]^2$ with Dirichlet data, using a sparse matrix (`scipy.sparse.lil_matrix`) and `spsolve`. The exact solution is $u = \sin x \sin y$. Results are saved for $h = 0.25, 0.2, 0.1, 0.01, 0.001$, together with a convergence plot (`Convergencerate_Five-points.py`).

### Finite elements — `FEM/`

A **P1 / constant-strain-triangle (CST)** finite element solver for $-\Delta u = 2\pi^2 \sin \pi x \sin \pi y$ on $[0,1]^2$, with $u = \sin \pi x \sin \pi y$ and homogeneous Dirichlet boundary conditions. It is built step by step:

- structured triangulation of the unit square and local-to-global node numbering;
- element stiffness matrices from the CST gradient formula, and load vectors by numerical quadrature;
- global assembly, then imposing the boundary conditions in one of two ways:
  - `FEM_Inner_Comparison.py`: solve only for interior nodes;
  - `FEM_CST_BCs_Comparison.py`: impose the boundary values on the full system;
- surface plots comparing the numerical and analytical solutions for $n = 4, 5, 10, 50, 100$ elements per side.

`FEM_CST_v4.py` is the latest stand-alone solver. `cst_v1.py`, `FEM_cst_v2.py`, `FEM_triangular_v3.py`, `help_3dv2.py`, `structure.py` and `code-terms.py` are earlier drafts and helper scripts, kept to show how the solver was developed.

## Running

```bash
pip install numpy scipy matplotlib pandas
cd Rungekuta && python 4order_convergence.py
cd ../FDM && python Convergencerate_Five-points.py
cd ../FEM && python FEM_CST_BCs_Comparison.py
```

Run the scripts from their own folders: the convergence scripts import the solver modules next to them (for example `from runge4order import *`).
