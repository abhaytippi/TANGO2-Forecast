"""IMEX solver for the stochastic reaction--diffusion crisis model.

Implements Algorithm 1 of the paper. Diffusion is treated implicitly for
unconditional stability (Proposition 1) and the reaction term explicitly so
that no nonlinear solve is needed:

``(I - dt D L) r^{n+1} = r^n + dt [alpha r^n (1 - r^n) + beta S - delta r^n] + xi^n``

with ``xi^n = sigma_eta sqrt(r^n (1 - r^n)) S dW^n`` and
``dW^n ~ N(0, dt I)`` handled by Euler--Maruyama.

Realisations are advanced in lockstep and the linear system is solved for all
of them in one LAPACK call, so an ensemble of 200 paths costs little more than
200 sequential single paths would in Python-loop terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .operators import TridiagonalOperator, build_implicit_operator
from .parameters import CrisisModelParameters, GridSpec

__all__ = ["ForecastResult", "IMEXCrisisSolver"]


@dataclass(slots=True)
class ForecastResult:
    """Output of a forecast run.

    All risk quantities are in units of the latent risk variable ``r``, which is
    dimensionless and defined only relative to the crisis threshold ``r_star``.

    Warning
    -------
    Crisis days recorded here are **model outputs, not validated predictions**.
    The model has never been compared against an observed crisis date; see the
    paper, Section 9. Use the regime classification relative to ``beta_critical``
    and the distribution of outcomes, not any individual day.
    """

    times: np.ndarray
    """Shape ``(n_steps + 1,)``, days since baseline assessment."""

    peak_risk: np.ndarray
    """Shape ``(n_steps + 1, n_realizations)``, ``max_x r(x, t)`` per path."""

    crisis_days: list[np.ndarray]
    """One array of crisis days per realisation."""

    x: np.ndarray
    """Spatial nodes."""

    parameters: CrisisModelParameters
    grid: GridSpec
    seed: int | None
    stochastic: bool

    field: np.ndarray | None = None
    """Shape ``(n_saved, N)``: full ``r(x, t)`` for realisation 0, if stored."""

    field_times: np.ndarray | None = None

    mass: np.ndarray | None = None
    """Shape ``(n_steps + 1, n_realizations)``: discrete ``integral of r dx``."""

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Summarise the run without dumping the underlying arrays."""
        return (
            f"ForecastResult(beta={self.parameters.beta:.5f}, "
            f"n_realizations={self.n_realizations}, "
            f"horizon_days={self.times[-1]:.0f}, "
            f"mean_crises={self.crisis_counts.mean():.2f}, "
            f"stochastic={self.stochastic}, seed={self.seed})"
        )

    @property
    def n_realizations(self) -> int:
        """Number of stochastic paths in this run."""
        return self.peak_risk.shape[1]

    @property
    def crisis_counts(self) -> np.ndarray:
        """Number of crises in the horizon, one entry per realisation."""
        return np.array([days.size for days in self.crisis_days], dtype=int)

    @property
    def first_crisis_days(self) -> np.ndarray:
        """Day of the first crisis per realisation; ``nan`` where none occurred."""
        return np.array(
            [days[0] if days.size else np.nan for days in self.crisis_days], dtype=float
        )

    @property
    def crisis_free_fraction(self) -> float:
        """Fraction of realisations with no crisis inside the horizon."""
        return float(np.mean(self.crisis_counts == 0))

    def peak_risk_quantiles(self, quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)):
        """Return ``(quantiles, values)`` of ``max_x r`` across realisations.

        ``values`` has shape ``(len(quantiles), n_steps + 1)`` and reproduces the
        median / interquartile / 5--95th percentile bands of paper Figure 6.
        """
        q = np.asarray(quantiles, dtype=float)
        return q, np.quantile(self.peak_risk, q, axis=1)

    def summary(self) -> dict[str, float]:
        """Ensemble summary statistics, matching those quoted in Section 5.5."""
        counts = self.crisis_counts.astype(float)
        first = self.first_crisis_days
        observed = first[~np.isnan(first)]
        return {
            "beta": self.parameters.beta,
            "n_realizations": float(self.n_realizations),
            "mean_crisis_count": float(counts.mean()),
            "sd_crisis_count": float(counts.std(ddof=1)) if counts.size > 1 else 0.0,
            "mean_first_crisis_day": float(observed.mean()) if observed.size else float("nan"),
            "sd_first_crisis_day": (
                float(observed.std(ddof=1)) if observed.size > 1 else float("nan")
            ),
            "probability_any_crisis": float(np.mean(counts > 0)),
            "final_peak_risk_median": float(np.median(self.peak_risk[-1])),
        }


class IMEXCrisisSolver:
    """Implicit--explicit integrator for Eq. (7) with crisis detection.

    Parameters
    ----------
    parameters
        Model parameters. Only ``beta`` is patient-specific.
    grid
        Spatial/temporal discretisation.

    Notes
    -----
    The implicit operator depends only on ``grid``, ``diffusivity`` and ``dt``,
    so it is assembled once per solver instance and reused across every call to
    :meth:`run`. Construct one solver and vary ``beta`` through
    :meth:`run(beta=...)` when sweeping a cohort.
    """

    def __init__(
        self,
        parameters: CrisisModelParameters | None = None,
        grid: GridSpec | None = None,
    ) -> None:
        self.parameters = parameters or CrisisModelParameters()
        self.grid = grid or GridSpec()
        self._x = self.grid.x
        self._source = self.parameters.source_profile(self._x)
        self._weights = self.grid.quadrature_weights
        self._operator: TridiagonalOperator = build_implicit_operator(
            self.grid, self.parameters.diffusivity, self.grid.dt_days
        )

    # ------------------------------------------------------------------
    @property
    def source(self) -> np.ndarray:
        """Gaussian vulnerability profile ``S(x)`` on the grid."""
        return self._source

    @property
    def x(self) -> np.ndarray:
        """Spatial nodes of the latent biomarker domain."""
        return self._x

    def run(
        self,
        beta: float | None = None,
        *,
        n_realizations: int = 1,
        seed: int | None = 0,
        stochastic: bool = True,
        enable_reset: bool = True,
        store_field: bool = False,
        field_stride: int = 1,
        track_mass: bool = False,
        initial_field: np.ndarray | None = None,
    ) -> ForecastResult:
        """Integrate the model over the horizon.

        Parameters
        ----------
        beta
            Source amplitude. Defaults to ``self.parameters.beta``.
        n_realizations
            Number of independent stochastic paths, advanced in lockstep.
        seed
            Seed for :func:`numpy.random.default_rng`. Recorded in the result so
            that any reported forecast is exactly reproducible.
        stochastic
            If ``False`` the noise term is dropped and the model is
            deterministic (used for the bifurcation analysis of Section 5.6).
        enable_reset
            If ``False`` the post-crisis reset of Eq. (6) is not applied.
            Crises are still recorded. Required for steady-state work, since
            ``beta_critical`` is defined with the reset disabled (Eq. 13).
        store_field
            Retain the full ``r(x, t)`` field for realisation 0 (paper Fig. 7).
        field_stride
            Store every ``field_stride``-th step to bound memory.
        track_mass
            Record the discrete spatial integral of ``r`` at every step. Used by
            the conservation test of Section 4.
        """
        params = self.parameters if beta is None else self.parameters.with_beta(beta)
        if beta is not None and self._source_needs_rebuild(params):  # pragma: no cover
            raise RuntimeError("source geometry changed; construct a new solver")

        n = self.grid.n_nodes
        n_steps = self.grid.n_steps
        dt = self.grid.dt_days
        k = int(n_realizations)
        if k < 1:
            raise ValueError("n_realizations must be >= 1")
        if stochastic and params.noise_intensity == 0.0:
            stochastic = False

        rng = np.random.default_rng(seed)
        source_col = self._source[:, None]

        if initial_field is None:
            r = np.full((n, k), params.r_initial, dtype=float)
        else:
            profile = np.asarray(initial_field, dtype=float)
            if profile.shape != (n,):
                raise ValueError(f"initial_field must have shape ({n},), got {profile.shape}")
            r = np.repeat(profile[:, None], k, axis=1)
        peak_risk = np.empty((n_steps + 1, k), dtype=float)
        peak_risk[0] = r.max(axis=0)
        crisis_days: list[list[float]] = [[] for _ in range(k)]

        mass = None
        if track_mass:
            mass = np.empty((n_steps + 1, k), dtype=float)
            mass[0] = self._mass(r)

        saved: list[np.ndarray] = []
        saved_times: list[float] = []
        if store_field:
            saved.append(r[:, 0].copy())
            saved_times.append(0.0)

        noise_scale = params.noise_intensity * np.sqrt(dt)

        for step in range(n_steps):
            reaction = params.alpha * r * (1.0 - r) + params.beta * source_col - params.delta * r
            rhs = r + dt * reaction

            if stochastic:
                # sqrt(r(1-r)) vanishes at r = 0 and r = 1, keeping r in [0, 1].
                envelope = np.sqrt(np.clip(r * (1.0 - r), 0.0, None))
                rhs += noise_scale * envelope * source_col * rng.standard_normal((n, k))

            r = self._operator.solve(rhs)
            np.clip(r, 0.0, 1.0, out=r)

            current_peak = r.max(axis=0)
            crossed = current_peak >= params.r_star
            if np.any(crossed):
                day = (step + 1) * dt
                for idx in np.flatnonzero(crossed):
                    crisis_days[idx].append(day)
                if enable_reset:
                    sub = r[:, crossed]
                    r[:, crossed] = params.r_rest + params.kappa * (sub - params.r_rest)
                    current_peak = r.max(axis=0)

            peak_risk[step + 1] = current_peak
            if mass is not None:
                mass[step + 1] = self._mass(r)
            if store_field and (step + 1) % field_stride == 0:
                saved.append(r[:, 0].copy())
                saved_times.append((step + 1) * dt)

        return ForecastResult(
            times=np.arange(n_steps + 1) * dt,
            peak_risk=peak_risk,
            crisis_days=[np.asarray(d, dtype=float) for d in crisis_days],
            x=self._x,
            parameters=params,
            grid=self.grid,
            seed=seed,
            stochastic=stochastic,
            field=np.asarray(saved) if store_field else None,
            field_times=np.asarray(saved_times) if store_field else None,
            mass=mass,
        )

    # ------------------------------------------------------------------
    def _mass(self, r: np.ndarray) -> np.ndarray:
        """Discrete spatial integral, per realisation.

        Uses the quadrature the chosen boundary treatment actually conserves;
        see :attr:`GridSpec.quadrature_weights`.
        """
        return self._weights @ r

    def _source_needs_rebuild(self, params: CrisisModelParameters) -> bool:
        return (
            params.x0 != self.parameters.x0
            or params.source_width != self.parameters.source_width
            or params.diffusivity != self.parameters.diffusivity
        )
