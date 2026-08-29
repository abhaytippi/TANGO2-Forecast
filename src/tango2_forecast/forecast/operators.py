"""Discrete operators for the IMEX scheme.

Contains the three-point Laplacian with no-flux (Neumann) boundaries and the
tridiagonal solver used for the implicit diffusion update.

The implicit matrix ``A = I - dt * D * L`` is constant in time, so it is
assembled once in LAPACK banded storage at construction. Each solve is ``O(N)``
and accepts a *batch* of right-hand sides, which is what makes ensemble
forecasting (many stochastic realisations advanced in lockstep) practical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .parameters import GridSpec

__all__ = [
    "TridiagonalOperator",
    "amplification_factor",
    "build_implicit_operator",
    "laplacian_bands",
    "thomas_solve",
]


def laplacian_bands(grid: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lower, diag, upper)`` bands of the three-point Laplacian.

    Interior rows use the standard second-order stencil
    ``(r[j-1] - 2 r[j] + r[j+1]) / dx**2``. Boundary rows enforce zero flux:

    ``one_sided``
        Ghost value equal to the boundary value (``r[-1] = r[0]``), giving the
        row ``(-r[0] + r[1]) / dx**2``. This is the paper's scheme. It is
        first-order at the boundary, which is why the observed global order in
        Table 2 is close to one despite the second-order interior stencil.

    ``mirrored``
        Reflected ghost value (``r[-1] = r[1]``), giving ``2 (r[1] - r[0]) / dx**2``,
        which is second-order.

    Both treatments conserve mass exactly: the column sums of ``L`` vanish, so
    ``sum(L @ r) == 0`` to round-off for any ``r``.
    """
    n = grid.n_nodes
    inv_dx2 = 1.0 / grid.dx**2

    lower = np.full(n - 1, inv_dx2)
    upper = np.full(n - 1, inv_dx2)
    diag = np.full(n, -2.0 * inv_dx2)

    if grid.boundary == "one_sided":
        diag[0] = -inv_dx2
        diag[-1] = -inv_dx2
    elif grid.boundary == "mirrored":
        upper[0] = 2.0 * inv_dx2
        lower[-1] = 2.0 * inv_dx2
    else:  # pragma: no cover - guarded by the pydantic Literal
        raise ValueError(f"unknown boundary treatment {grid.boundary!r}")

    return lower, diag, upper


def thomas_solve(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal system by the Thomas algorithm in ``O(N)``.

    This is the algorithm named in the paper (Algorithm 1, line 7).

    A transparent implementation of the algorithm named in the paper. The
    production path uses :class:`TridiagonalOperator`, which dispatches to
    LAPACK; the two are checked against each other in the test suite.
    """
    n = diag.size
    c_prime = np.empty(n - 1)
    d_prime = np.empty(n)

    pivot = diag[0]
    if pivot == 0.0:
        raise ZeroDivisionError("tridiagonal matrix is singular at row 0")
    c_prime[0] = upper[0] / pivot
    d_prime[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diag[i] - lower[i - 1] * c_prime[i - 1]
        if pivot == 0.0:
            raise ZeroDivisionError(f"tridiagonal matrix is singular at row {i}")
        if i < n - 1:
            c_prime[i] = upper[i] / pivot
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot

    x = np.empty(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


@dataclass(frozen=True, slots=True)
class TridiagonalOperator:
    """A constant tridiagonal matrix in LAPACK banded storage.

    Attributes
    ----------
    lower, diag, upper
        The three bands, retained for inspection and for the reference solver.
    banded
        Shape ``(3, N)`` array in :func:`scipy.linalg.solve_banded` layout.
    """

    lower: np.ndarray
    diag: np.ndarray
    upper: np.ndarray
    banded: np.ndarray

    @classmethod
    def build(cls, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> TridiagonalOperator:
        """Assemble the operator from its three bands."""
        n = diag.size
        banded = np.zeros((3, n))
        banded[0, 1:] = upper
        banded[1, :] = diag
        banded[2, :-1] = lower
        return cls(lower, diag, upper, banded)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``A x = rhs``.

        ``rhs`` may have shape ``(N,)`` or ``(N, K)``; in the latter case ``K``
        independent systems are solved in a single LAPACK call.
        """
        return solve_banded((1, 1), self.banded, rhs, check_finite=False)

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply the matrix to ``v`` (shape ``(N,)``). Used for residual checks."""
        out = self.diag * v
        out[:-1] += self.upper * v[1:]
        out[1:] += self.lower * v[:-1]
        return out

    def is_diagonally_dominant(self) -> bool:
        """Weak diagonal dominance: sufficient for stable Gaussian elimination."""
        off = np.zeros_like(self.diag)
        off[:-1] += np.abs(self.upper)
        off[1:] += np.abs(self.lower)
        return bool(np.all(np.abs(self.diag) >= off))


def build_implicit_operator(
    grid: GridSpec, diffusivity: float, dt_days: float
) -> TridiagonalOperator:
    """Assemble ``A = I - dt * D * L`` for the implicit diffusion update.

    ``A`` is tridiagonal and strictly diagonally dominant for every ``dt > 0``
    because ``D > 0`` -- the discrete counterpart of Proposition 1.
    """
    lower, diag, upper = laplacian_bands(grid)
    scale = -dt_days * diffusivity
    return TridiagonalOperator.build(scale * lower, 1.0 + scale * diag, scale * upper)


def amplification_factor(
    grid: GridSpec, diffusivity: float, dt_days: float, theta: np.ndarray
) -> np.ndarray:
    """Von Neumann amplification factor of the implicit diffusive update.

    Substituting ``r[j]^n = g^n exp(i theta j)`` into the homogeneous part of
    Eq. (10) gives ``g(theta) = 1 / (1 + 4 lam sin^2(theta / 2))`` with
    ``lam = D dt / dx^2 > 0`` (Proposition 1). The denominator is at least one,
    so ``0 < g <= 1`` for every ``theta`` and no mode is amplified.
    """
    lam = diffusivity * dt_days / grid.dx**2
    return 1.0 / (1.0 + 4.0 * lam * np.sin(theta / 2.0) ** 2)
