"""Exact solver used for every PDE result in the IJSCAR manuscript.

    Tippimath, A. (2026). Dynamic Mathematical PDE Modeling and Machine Learning
    for TANGO2 Identification and Crisis Prediction. IJSCAR 1(1).

This file is deliberately small and self contained so that a reader can check it
line by line against Section 4 of the paper. It does not depend on the
interactive application in ``src/``.

Model (paper Eq. 1 and the reset rule)
    dr/dt = D r_xx + alpha r(1-r) + beta S(x) - delta r
            + sigma_eta sqrt(r(1-r)) S(x) dW
    S(x)  = exp(-(x - x0)^2 / (2 sigma^2))
    crisis when max_x r >= r_star, then r <- r_rest + kappa (r - r_rest)

Discretisation
    N = 260 nodes on x = linspace(0, 1, N), dx = 1/N, no flux boundaries
    (finite volume rows), IMEX Euler: diffusion implicit (tridiagonal,
    Thomas algorithm via scipy.linalg.solve_banded), reaction explicit,
    Euler-Maruyama noise. dt = 0.5 d, T = 900 d, r(x, 0) = 0.13.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class Params:
    D: float = 0.0012          # diffusion coefficient
    alpha: float = 0.011       # logistic growth rate (1/day)
    delta: float = 0.007       # clearance rate (1/day)
    sigma: float = 0.06        # source width
    x0: float = 0.58           # vulnerability locus
    r_star: float = 0.72       # crisis threshold
    kappa: float = 0.52        # post crisis retention
    r_rest: float = 0.0        # resting level used by the reset (paper: 0)
    sigma_eta: float = 0.045   # noise intensity
    r0: float = 0.13           # uniform initial condition
    N: int = 260               # spatial nodes
    dt: float = 0.5            # time step (days)
    T: float = 900.0           # horizon (days)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


BASE = Params()
BETA_MIN, BETA_MAX = 0.0121, 0.0176


def grid(p: Params, dt: float | None = None):
    """Node coordinates and the banded implicit diffusion matrix."""
    dt = p.dt if dt is None else dt
    dx = 1.0 / p.N
    x = np.linspace(0.0, 1.0, p.N)
    lam = p.D * dt / dx**2
    ab = np.zeros((3, p.N))
    ab[0, 1:] = -lam
    ab[2, :-1] = -lam
    ab[1, :] = 1.0 + 2.0 * lam
    ab[1, 0] = ab[1, -1] = 1.0 + lam  # no flux boundary rows
    return x, ab


def source(p: Params, x: np.ndarray) -> np.ndarray:
    return np.exp(-((x - p.x0) ** 2) / (2.0 * p.sigma**2))


def simulate(beta: float, n_real: int, seed: int, p: Params = BASE, *, noise: bool = True,
             reset: bool = True):
    """Run ``n_real`` independent realisations (vectorised).

    Returns (crisis_count[n_real], first_crisis_day[n_real]); the first crisis day
    is NaN for realisations with no crisis.
    """
    x, ab = grid(p)
    S = source(p, x)[:, None]
    rng = np.random.default_rng(seed)
    nt = int(p.T / p.dt)
    r = np.full((p.N, n_real), p.r0)
    count = np.zeros(n_real)
    first = np.full(n_real, np.nan)
    for n in range(nt):
        rhs = r + p.dt * (p.alpha * r * (1 - r) + beta * S - p.delta * r)
        if noise and p.sigma_eta > 0:
            rhs = rhs + p.sigma_eta * np.sqrt(r * (1 - r)) * S * rng.normal(0.0, np.sqrt(p.dt), (p.N, n_real))
        r = np.clip(solve_banded((1, 1), ab, rhs), 0.0, 1.0)
        hit = r.max(axis=0) >= p.r_star
        if hit.any():
            count[hit] += 1
            first[hit & np.isnan(first)] = (n + 1) * p.dt
            if reset:
                r[:, hit] = p.r_rest + p.kappa * (r[:, hit] - p.r_rest)
    return count, first


def scalar_model(b_eff: float, n_real: int, seed: int, noise_eff: float, p: Params = BASE):
    """Scalar reduction without diffusion (same threshold, reset and noise form)."""
    rng = np.random.default_rng(seed)
    nt = int(p.T / p.dt)
    r = np.full(n_real, p.r0)
    count = np.zeros(n_real)
    for _ in range(nt):
        r = np.clip(r + p.dt * (p.alpha * r * (1 - r) + b_eff - p.delta * r)
                    + noise_eff * np.sqrt(r * (1 - r)) * rng.normal(0.0, np.sqrt(p.dt), n_real), 0.0, 1.0)
        hit = r >= p.r_star
        count[hit] += 1
        r[hit] = p.r_rest + p.kappa * (r[hit] - p.r_rest)
    return count


def steady_peak(beta: float, p: Params = BASE, n_steps: int = 40000) -> float:
    """Deterministic steady state peak with the reset disabled."""
    x, ab = grid(p)
    S = source(p, x)
    r = np.full(p.N, p.r0)
    for _ in range(n_steps):
        r = np.clip(solve_banded((1, 1), ab, r + p.dt * (p.alpha * r * (1 - r) + beta * S - p.delta * r)), 0.0, 1.0)
    return float(r.max())


def critical_amplitude(p: Params = BASE, lo: float = 0.002, hi: float = 0.04, iters: int = 40) -> float:
    """beta* = inf{beta : steady peak >= r_star}, by bisection (reset disabled)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if steady_peak(mid, p) >= p.r_star:
            hi = mid
        else:
            lo = mid
    return hi


def deterministic_field(beta: float, dt: float, T: float, p: Params = BASE) -> np.ndarray:
    """Deterministic field at time T (no noise, no reset), for convergence tests."""
    x, ab = grid(p, dt)
    S = source(p, x)
    r = np.full(p.N, p.r0)
    for _ in range(int(round(T / dt))):
        r = np.clip(solve_banded((1, 1), ab, r + dt * (p.alpha * r * (1 - r) + beta * S - p.delta * r)), 0.0, 1.0)
    return r
