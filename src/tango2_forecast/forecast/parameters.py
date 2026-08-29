"""Parameter objects for the stochastic reaction--diffusion crisis model.

Every default in this module is taken from Table 1 of Tippimath, Lalani & Liu,
*A Hybrid Random Forest and Reaction--Diffusion Framework for Early
Identification and Crisis Forecasting in TANGO2 Deficiency Disorder*.

Provenance of each value is recorded in the field description so that a reader
can distinguish parameters fixed by clinical information from those obtained by
manual calibration. **No parameter in this module was fitted to observed crisis
dates**; the source cohort contains none. See :mod:`tango2_forecast.disclaimer`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["BETA_CRITICAL_PUBLISHED", "BETA_MAX", "BETA_MIN", "CrisisModelParameters", "GridSpec"]

#: Calibrated patient-specific source-amplitude range (paper, Table 1).
BETA_MIN: float = 0.0121
BETA_MAX: float = 0.0176

#: Critical source amplitude reported in the paper, Eq. (14). The value computed
#: by :func:`tango2_forecast.forecast.steady_state.critical_amplitude` is
#: regression-tested against this constant.
BETA_CRITICAL_PUBLISHED: float = 0.01273


class GridSpec(BaseModel):
    """Spatial and temporal discretisation of the latent biomarker domain.

    The biomarker coordinate ``x`` indexes coupled physiological subsystems and
    is dimensionless on ``[0, 1]``. It has no measured physiological anchor; see
    the paper, Section 3.3 and Section 10 ("Grounding the latent coordinate").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_nodes: int = Field(
        default=260,
        ge=8,
        le=20_000,
        description="Number of spatial nodes N (paper: 260).",
    )
    dt_days: float = Field(
        default=0.5,
        gt=0.0,
        le=30.0,
        description="Time step in days (paper: 0.5 d).",
    )
    horizon_days: float = Field(
        default=900.0,
        gt=0.0,
        le=20_000.0,
        description="Integration horizon T in days (paper: 900 d).",
    )
    boundary: Literal["one_sided", "mirrored"] = Field(
        default="one_sided",
        description=(
            "Discrete no-flux treatment. 'one_sided' reproduces the paper's "
            "scheme (first-order boundary rows, which dominate the observed "
            "convergence order); 'mirrored' uses ghost-node reflection and is "
            "second-order. Both conserve mass exactly under pure diffusion."
        ),
    )

    @property
    def dx(self) -> float:
        """Grid spacing, ``1 / (N - 1)``."""
        return 1.0 / (self.n_nodes - 1)

    @property
    def n_steps(self) -> int:
        """Number of time steps to reach the horizon."""
        return round(self.horizon_days / self.dt_days)

    @property
    def x(self) -> np.ndarray:
        """Node coordinates on ``[0, 1]``."""
        return np.linspace(0.0, 1.0, self.n_nodes)

    @property
    def quadrature_weights(self) -> np.ndarray:
        """Weights ``w`` such that ``w @ r`` is the discretely conserved integral.

        The two no-flux treatments conserve *different* discrete measures, and
        using the wrong one turns an exactly conserved quantity into one that
        drifts at truncation-error level:

        * ``one_sided`` rows annihilate the plain column sum, so the conserved
          measure is the rectangle rule ``dx * sum(r)``.
        * ``mirrored`` rows annihilate the trapezoidal column sum, so the
          conserved measure is ``dx * (r[0]/2 + r[1:-1] + r[-1]/2)``.

        In both cases ``sum(w * (L @ r)) == 0`` to round-off for every ``r``.
        """
        w = np.full(self.n_nodes, self.dx)
        if self.boundary == "mirrored":
            w[0] *= 0.5
            w[-1] *= 0.5
        return w

    @property
    def diffusion_number_limit(self) -> float:
        """``dx**2 / 2`` -- the explicit stability limit divided by ``D``.

        Multiplying by ``1 / D`` gives the time step an explicit scheme would be
        restricted to (paper, Eq. 9). The IMEX scheme is not subject to it.
        """
        return 0.5 * self.dx**2


class CrisisModelParameters(BaseModel):
    """Parameters of the stochastic reaction--diffusion model, Eq. (4) and (7).

    Only :attr:`beta` varies between patients; it is set from the baseline
    phenotype burden by :func:`tango2_forecast.forecast.severity.beta_from_severity`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- Patient-specific -------------------------------------------------
    beta: float = Field(
        default=0.0158,
        ge=0.0,
        le=1.0,
        description="Source amplitude. The only patient-specific parameter.",
    )

    # --- Fixed by clinical information -----------------------------------
    r_star: float = Field(
        default=0.72,
        gt=0.0,
        le=1.0,
        description="Crisis threshold. Fixes the scale of the latent risk variable.",
    )
    kappa: float = Field(
        default=0.52,
        ge=0.0,
        le=1.0,
        description="Post-crisis retention: fraction of supra-resting risk retained.",
    )

    # --- Manually calibrated to qualitative published behaviour -----------
    alpha: float = Field(default=0.011, ge=0.0, description="Logistic growth rate (1/day).")
    delta: float = Field(default=0.007, ge=0.0, description="Clearance rate (1/day).")
    diffusivity: float = Field(
        default=0.0012,
        gt=0.0,
        description="Diffusion coefficient D, propagation between subsystems.",
    )
    x0: float = Field(default=0.58, ge=0.0, le=1.0, description="Vulnerability locus.")
    source_width: float = Field(
        default=0.06, gt=0.0, le=1.0, description="Gaussian source width sigma."
    )
    noise_intensity: float = Field(
        default=0.045, ge=0.0, description="Multiplicative noise intensity sigma_eta."
    )

    # --- Initial / resting state ------------------------------------------
    r_initial: float = Field(
        default=0.13, ge=0.0, le=1.0, description="Uniform initial risk r(x, 0)."
    )
    r_rest: float = Field(
        default=0.13,
        ge=0.0,
        le=1.0,
        description=(
            "Resting risk floor used by the post-crisis reset, Eq. (6). The "
            "paper does not tabulate this separately; it is taken equal to the "
            "initial condition r_0."
        ),
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> CrisisModelParameters:
        if self.delta <= 0.0 and self.alpha > 0.0:
            raise ValueError(
                "clearance rate delta must be strictly positive whenever there is "
                "logistic growth: without clearance no bounded sub-threshold state "
                "exists (paper, Section 3.5). delta = 0 is permitted only in the "
                "degenerate alpha = 0 case used by the pure-diffusion "
                "conservation test."
            )
        if self.r_rest > self.r_star:
            raise ValueError("r_rest must lie below the crisis threshold r_star.")
        if self.r_initial > self.r_star:
            raise ValueError(
                "r_initial must lie below r_star, otherwise a crisis is declared at t = 0."
            )
        return self

    # ------------------------------------------------------------------
    def source_profile(self, x: np.ndarray) -> np.ndarray:
        """Gaussian vulnerability source ``S(x)``, peak-normalised to 1.

        ``S(x) = exp[-(x - x0)^2 / (2 sigma^2)]`` (paper, Section 3.4). The
        profile is normalised to unit *peak*, not unit mass, which is the
        convention Eq. (12) assumes.
        """
        return np.exp(-((x - self.x0) ** 2) / (2.0 * self.source_width**2))

    def well_mixed_fixed_point(self) -> float:
        """Stable fixed point of the well-mixed reduction, Eq. (12).

        Returns ``r+ = [(alpha - delta) + sqrt((alpha - delta)^2 + 4 alpha beta)] / (2 alpha)``.

        Note
        ----
        This uses the *peak* source amplitude ``beta`` rather than its spatial
        average, which is the convention under which the paper's claim that
        ``r+ > 1`` for every calibrated ``beta`` holds. Under an area-averaged
        source the reduction has a bounded fixed point, so the identity of the
        quiescent regime with a purely spatial effect is convention-dependent.
        The value returned here is therefore an upper bound on the well-mixed
        steady state, and is reported as such in the documentation.
        """
        gap = self.alpha - self.delta
        return (gap + np.sqrt(gap**2 + 4.0 * self.alpha * self.beta)) / (2.0 * self.alpha)

    def with_beta(self, beta: float) -> CrisisModelParameters:
        """Return a copy with a new source amplitude."""
        return self.model_copy(update={"beta": beta})

    def perturbed(self, name: str, fraction: float) -> CrisisModelParameters:
        """Return a copy with field ``name`` scaled by ``1 + fraction``.

        Used by the elasticity analysis of paper Section 5.7.
        """
        if name not in type(self).model_fields:
            raise KeyError(f"unknown parameter {name!r}")
        current = getattr(self, name)
        return self.model_copy(update={name: current * (1.0 + fraction)})
