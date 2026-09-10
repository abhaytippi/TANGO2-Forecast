"""Numerical verification of the IMEX solver (paper, Section 4).

These routines establish that *the equations are being solved correctly*. They
say nothing about whether the model is biologically right, which is a separate
claim addressed in the paper's Section 9. The distinction is deliberate and is
preserved throughout this package: the model is **mathematically verified, not
biologically validated**.

Three checks are implemented:

``spatial_convergence`` / ``temporal_convergence``
    Refinement studies against a fine reference solution. First order is the
    expected outcome, since the one-sided Neumann boundary rows dominate the
    second-order interior stencil.

``mass_conservation_drift``
    With ``alpha = delta = beta = 0`` the model reduces to pure diffusion with
    no-flux boundaries, for which the spatial integral of ``r`` is exactly
    conserved. A sign error in the boundary rows shows up immediately as drift.

``max_amplification``
    von Neumann check against Proposition 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

from .operators import amplification_factor
from .parameters import CrisisModelParameters, GridSpec
from .solver import IMEXCrisisSolver

__all__ = [
    "RefinementLevel",
    "mass_conservation_drift",
    "max_amplification",
    "spatial_convergence",
    "temporal_convergence",
]


@dataclass(frozen=True, slots=True)
class RefinementLevel:
    """One row of a refinement study."""

    resolution: float
    """``N`` for spatial studies, ``dt`` in days for temporal studies."""

    error: float
    order: float | None
    """Observed order of accuracy against the previous, coarser level."""


def _final_field(
    params: CrisisModelParameters, grid: GridSpec, horizon_days: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, reset-free final state ``(x, r(x, T))``."""
    grid = grid.model_copy(update={"horizon_days": horizon_days})
    solver = IMEXCrisisSolver(params, grid)
    result = solver.run(
        n_realizations=1,
        stochastic=False,
        enable_reset=False,
        store_field=True,
        field_stride=grid.n_steps,
        seed=None,
    )
    assert result.field is not None
    return grid.x, result.field[-1]


def _relative_l2(
    coarse_x: np.ndarray, coarse_r: np.ndarray, ref_x: np.ndarray, ref_r: np.ndarray
) -> float:
    """Relative L2 error after interpolating the coarse solution to the reference grid."""
    interpolated = np.interp(ref_x, coarse_x, coarse_r)
    return float(np.linalg.norm(interpolated - ref_r) / np.linalg.norm(ref_r))


def _orders(errors: list[float], refinement: float = 2.0) -> list[float | None]:
    out: list[float | None] = [None]
    for prev, cur in pairwise(errors):
        out.append(float(np.log(prev / cur) / np.log(refinement)))
    return out


def spatial_convergence(
    node_counts: tuple[int, ...] = (65, 130, 260, 520, 1040),
    reference_nodes: int = 2080,
    *,
    params: CrisisModelParameters | None = None,
    horizon_days: float = 100.0,
    dt_days: float = 0.5,
    boundary: Literal["one_sided", "mirrored"] = "one_sided",
) -> list[RefinementLevel]:
    """Spatial refinement study against a fine reference (paper, Table 2, left)."""
    params = params or CrisisModelParameters()
    base = GridSpec(n_nodes=reference_nodes, dt_days=dt_days, boundary=boundary)
    ref_x, ref_r = _final_field(params, base, horizon_days)

    errors: list[float] = []
    for n in node_counts:
        grid = GridSpec(n_nodes=n, dt_days=dt_days, boundary=boundary)
        x, r = _final_field(params, grid, horizon_days)
        errors.append(_relative_l2(x, r, ref_x, ref_r))

    return [
        RefinementLevel(float(n), e, o)
        for n, e, o in zip(node_counts, errors, _orders(errors), strict=True)
    ]


def temporal_convergence(
    time_steps: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625),
    reference_dt: float = 0.015625,
    *,
    params: CrisisModelParameters | None = None,
    n_nodes: int = 260,
    horizon_days: float = 100.0,
    boundary: Literal["one_sided", "mirrored"] = "one_sided",
) -> list[RefinementLevel]:
    """Temporal refinement study (paper, Table 2, right).

    First order is expected: the IMEX splitting is first-order in time, as is
    Euler--Maruyama in the stochastic case.
    """
    params = params or CrisisModelParameters()
    ref_grid = GridSpec(n_nodes=n_nodes, dt_days=reference_dt, boundary=boundary)
    ref_x, ref_r = _final_field(params, ref_grid, horizon_days)

    errors: list[float] = []
    for dt in time_steps:
        grid = GridSpec(n_nodes=n_nodes, dt_days=dt, boundary=boundary)
        x, r = _final_field(params, grid, horizon_days)
        errors.append(_relative_l2(x, r, ref_x, ref_r))

    return [
        RefinementLevel(dt, e, o)
        for dt, e, o in zip(time_steps, errors, _orders(errors), strict=True)
    ]


def mass_conservation_drift(
    *,
    n_nodes: int = 260,
    dt_days: float = 0.5,
    n_steps: int = 2000,
    boundary: Literal["one_sided", "mirrored"] = "one_sided",
) -> float:
    """Relative drift of the spatial integral under pure diffusion.

    Sets ``alpha = delta = beta = 0`` so that Eq. (4) reduces to the heat
    equation with no-flux boundaries. Returns ``|M(T) - M(0)| / M(0)``, which
    should sit at double-precision round-off.
    """
    params = CrisisModelParameters(
        alpha=0.0, delta=0.0, beta=0.0, noise_intensity=0.0, r_initial=0.4, r_star=1.0
    )
    grid = GridSpec(
        n_nodes=n_nodes, dt_days=dt_days, horizon_days=n_steps * dt_days, boundary=boundary
    )
    solver = IMEXCrisisSolver(params, grid)

    # Start from a strongly non-uniform profile so diffusion actually does work.
    # A flat profile would conserve mass trivially and test nothing.
    profile = 0.2 + 0.6 * np.exp(-((grid.x - 0.35) ** 2) / (2 * 0.05**2))

    result = solver.run(
        n_realizations=1,
        stochastic=False,
        enable_reset=False,
        track_mass=True,
        seed=None,
        initial_field=profile,
    )
    assert result.mass is not None
    m0, m1 = result.mass[0, 0], result.mass[-1, 0]
    return float(abs(m1 - m0) / abs(m0))


def max_amplification(
    *, n_nodes: int = 260, dt_days: float = 0.5, diffusivity: float = 0.0012, n_theta: int = 4001
) -> float:
    """Maximum von Neumann amplification factor over all wavenumbers."""
    grid = GridSpec(n_nodes=n_nodes, dt_days=dt_days)
    theta = np.linspace(-np.pi, np.pi, n_theta)
    return float(np.max(amplification_factor(grid, diffusivity, dt_days, theta)))
