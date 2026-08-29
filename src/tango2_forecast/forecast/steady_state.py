"""Steady states of the deterministic model and the critical amplitude beta*.

The critical source amplitude is defined in Eq. (13) as

``beta* = inf { beta > 0 : max_x r_inf(x; beta) >= r* }``

where ``r_inf`` solves the *reset-disabled* steady-state problem

``0 = D r'' + alpha r (1 - r) + beta S - delta r``

with no-flux boundaries. The paper obtains ``beta* = 0.01273`` by bisection.

Two solvers are provided. :func:`steady_state` uses a damped Newton iteration on
the discretised residual, which converges quadratically and costs a handful of
tridiagonal solves; :func:`critical_amplitude` brackets and bisects on the peak
of that solution, warm-starting each Newton solve from the previous one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .operators import build_implicit_operator, laplacian_bands
from .parameters import CrisisModelParameters, GridSpec

__all__ = ["SteadyState", "bifurcation_curve", "critical_amplitude", "steady_state"]


class SteadyStateConvergenceError(RuntimeError):
    """Raised when the Newton iteration fails to converge."""


@dataclass(frozen=True, slots=True)
class SteadyState:
    """A converged steady-state profile."""

    x: np.ndarray
    r: np.ndarray
    beta: float
    residual_norm: float
    iterations: int

    @property
    def peak(self) -> float:
        """Maximum risk over the domain, the quantity compared against ``r_star``."""
        return float(self.r.max())

    def exceeds(self, r_star: float) -> bool:
        """Whether this steady state reaches the crisis threshold."""
        return self.peak >= r_star


def _residual_and_jacobian_bands(
    r: np.ndarray,
    params: CrisisModelParameters,
    source: np.ndarray,
    lap: tuple[np.ndarray, np.ndarray, np.ndarray],
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual ``F(r)`` and the banded Jacobian ``dF/dr``.

    ``F(r) = D L r + alpha r (1 - r) + beta S - delta r`` and
    ``J = D L + diag(alpha (1 - 2 r) - delta)``, which is tridiagonal because
    ``L`` is and the reaction term is pointwise.
    """
    lower, diag, upper = lap
    d = params.diffusivity

    lap_r = diag * r
    lap_r[:-1] += upper * r[1:]
    lap_r[1:] += lower * r[:-1]

    residual = d * lap_r + params.alpha * r * (1.0 - r) + beta * source - params.delta * r

    n = r.size
    banded = np.zeros((3, n))
    banded[0, 1:] = d * upper
    banded[1, :] = d * diag + params.alpha * (1.0 - 2.0 * r) - params.delta
    banded[2, :-1] = d * lower
    return residual, banded


def _pseudo_transient(
    params: CrisisModelParameters,
    grid: GridSpec,
    source: np.ndarray,
    beta: float,
    *,
    dt: float = 10.0,
    n_steps: int = 2000,
) -> np.ndarray:
    """Relax towards the steady state by long IMEX time stepping.

    The reaction term is a downward parabola in ``r`` and therefore has two
    roots. A cold Newton start from the uniform initial condition converges to
    the *unstable* lower root, which is not the state the dynamics reach. This
    routine integrates the deterministic model with a large time step, without
    the crisis reset, to land in the basin of the stable branch; Newton then
    polishes the result to machine precision.

    The relaxation time is of order ``1 / |f'(r+)| ~ 100`` days, so the default
    of 20 000 days is ample.
    """
    operator = build_implicit_operator(grid, params.diffusivity, dt)
    r = np.full(grid.n_nodes, params.r_initial)
    for _ in range(n_steps):
        reaction = params.alpha * r * (1.0 - r) + beta * source - params.delta * r
        r = np.clip(operator.solve(r + dt * reaction), 0.0, 1.0)
    return r


def steady_state(
    params: CrisisModelParameters,
    grid: GridSpec | None = None,
    *,
    beta: float | None = None,
    initial_guess: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> SteadyState:
    """Solve the reset-disabled steady-state problem, Eq. (13).

    Uses pseudo-transient continuation to reach the stable branch followed by a
    damped Newton polish. The reset of Eq. (6) is deliberately not applied: a
    reset acts only after a threshold crossing and so cannot influence a steady
    state, which is the internal consistency check reported in Section 5.7
    (``kappa`` leaves ``beta*`` exactly unchanged).

    Parameters
    ----------
    initial_guess
        Warm start, e.g. the solution at a neighbouring ``beta``. When omitted,
        a pseudo-transient relaxation is run instead.
    tol
        Convergence tolerance on ``max |F(r)|``.

    Raises
    ------
    SteadyStateConvergenceError
        If the iteration does not converge within ``max_iter`` steps.
    """
    grid = grid or GridSpec()
    beta = params.beta if beta is None else beta
    x = grid.x
    source = params.source_profile(x)
    lap = laplacian_bands(grid)

    def newton(r: np.ndarray) -> SteadyState:
        for iteration in range(1, max_iter + 1):
            residual, jac = _residual_and_jacobian_bands(r, params, source, lap, beta)
            norm = float(np.max(np.abs(residual)))
            if norm < tol:
                return SteadyState(x, r, beta, norm, iteration - 1)

            step = solve_banded((1, 1), jac, -residual, check_finite=False)

            # Damped line search: take the largest step in {1, 1/2, 1/4, ...}
            # that decreases the residual norm and keeps the iterate physical.
            damping = 1.0
            for _ in range(40):
                trial = np.clip(r + damping * step, 0.0, 1.0)
                trial_residual, _ = _residual_and_jacobian_bands(trial, params, source, lap, beta)
                if np.max(np.abs(trial_residual)) < norm:
                    break
                damping *= 0.5
            else:
                raise SteadyStateConvergenceError(
                    f"line search stalled at iteration {iteration}, residual {norm:.3e}"
                )
            r = trial

        raise SteadyStateConvergenceError(
            f"Newton did not converge in {max_iter} iterations (residual {norm:.3e})"
        )

    if initial_guess is not None:
        try:
            return newton(initial_guess.copy())
        except SteadyStateConvergenceError:
            # The warm start was too far from the branch at this beta; fall back
            # to relaxation, which is slower but has a much larger basin.
            pass
    return newton(_pseudo_transient(params, grid, source, beta))


def critical_amplitude(
    params: CrisisModelParameters | None = None,
    grid: GridSpec | None = None,
    *,
    lower: float = 1e-4,
    upper: float = 0.05,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Compute ``beta*`` by bisection on the steady-state peak, Eq. (13).

    Returns the smallest ``beta`` for which the reset-disabled steady state
    reaches the crisis threshold. Newton solves are warm-started along the
    bracket, so the whole computation is a few hundred tridiagonal solves.

    Raises
    ------
    ValueError
        If the initial bracket does not straddle the threshold.
    """
    params = params or CrisisModelParameters()
    grid = grid or GridSpec()

    def peak(beta: float, guess: np.ndarray | None) -> tuple[float, np.ndarray | None]:
        """Steady-state peak at ``beta``.

        If no bounded steady state exists -- which happens once the source
        overwhelms clearance and diffusive leakage, so that ``r`` saturates at
        the carrying capacity -- the state is unambiguously supra-threshold and
        ``inf`` is returned. Treating non-convergence as "above threshold"
        rather than as an error is what lets the bracket be set generously.
        """
        try:
            sol = steady_state(params, grid, beta=beta, initial_guess=guess)
        except SteadyStateConvergenceError:
            return float("inf"), None
        return sol.peak, sol.r

    peak_lo, guess_lo = peak(lower, None)
    peak_hi, _ = peak(upper, guess_lo)
    if peak_lo >= params.r_star:
        raise ValueError(f"lower bracket {lower} already exceeds the threshold")
    if peak_hi < params.r_star:
        raise ValueError(f"upper bracket {upper} does not reach the threshold")

    guess = guess_lo
    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        peak_mid, new_guess = peak(mid, guess)
        if new_guess is not None:
            guess = new_guess
        if peak_mid >= params.r_star:
            upper = mid
        else:
            lower = mid
        if upper - lower < tol:
            break
    return 0.5 * (lower + upper)


def bifurcation_curve(
    params: CrisisModelParameters | None = None,
    grid: GridSpec | None = None,
    *,
    beta_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state peak risk as a function of ``beta`` (paper, Figure 8a).

    Returns ``(beta_values, peak_risk)``. Solves are warm-started along the
    sweep, which both speeds it up and keeps the branch continuous.
    """
    params = params or CrisisModelParameters()
    grid = grid or GridSpec()
    if beta_values is None:
        beta_values = np.linspace(0.008, 0.020, 61)

    peaks = np.empty(beta_values.size)
    guess: np.ndarray | None = None
    for i, beta in enumerate(beta_values):
        sol = steady_state(params, grid, beta=float(beta), initial_guess=guess)
        peaks[i] = sol.peak
        guess = sol.r
    return np.asarray(beta_values, dtype=float), peaks
