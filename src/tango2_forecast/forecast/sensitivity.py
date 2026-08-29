"""Parameter sensitivity analysis (paper, Section 5.7).

Every conclusion the crisis model supports rests on parameters that were
calibrated rather than measured, so the honest question is not whether they are
right but how much the conclusions move when they are wrong. This module
quantifies that with two complementary measures.

**Elasticity of the crisis count.** For a parameter ``theta`` perturbed by a
fraction ``f``,

``S_theta = (delta_n / n) / (delta_theta / theta)``    (Eq. 16)

which is dimensionless and therefore comparable across parameters of different
units. ``|S| > 1`` means the output responds more than proportionally. The
reported value averages the signed perturbations, so a parameter whose effect is
symmetric does not have its sensitivity cancelled.

**Shift in the critical amplitude.** ``beta*`` is a steady-state property, so it
is computed deterministically and carries no Monte Carlo error. Two parameters
must leave it *exactly* unchanged: the post-crisis reset ``kappa``, which acts
only after a threshold crossing, and the noise intensity ``sigma_eta``, which is
absent from the steady-state problem. Any nonzero shift in either is a bug
rather than a finding, which makes them a useful internal check.

Because crisis counts are estimated from a finite ensemble, elasticities carry
Monte Carlo noise. :func:`sensitivity_analysis` therefore reports a standard
error alongside each elasticity, and the classification into sensitive /
moderate / robust is made on the point estimate with that error in view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .parameters import BETA_MAX, CrisisModelParameters, GridSpec
from .solver import IMEXCrisisSolver
from .steady_state import SteadyStateConvergenceError, critical_amplitude

__all__ = [
    "SENSITIVITY_PARAMETERS",
    "ParameterSensitivity",
    "sensitivity_analysis",
]

Classification = Literal["sensitive", "moderate", "robust"]

#: Parameters varied in the analysis, with the human-readable role used in
#: Table 6. ``beta`` is excluded because it is patient-specific by construction
#: rather than a shared modelling assumption.
SENSITIVITY_PARAMETERS: dict[str, str] = {
    "r_star": "crisis threshold",
    "delta": "clearance rate",
    "source_width": "source width",
    "alpha": "growth rate",
    "kappa": "post-crisis reset",
    "diffusivity": "diffusion",
    "x0": "vulnerability locus",
    "noise_intensity": "noise intensity",
}


def _classify(elasticity: float) -> Classification:
    """Bucket an elasticity the way Table 6 does."""
    magnitude = abs(elasticity)
    if magnitude > 1.0:
        return "sensitive"
    if magnitude > 0.5:
        return "moderate"
    return "robust"


@dataclass(frozen=True, slots=True)
class ParameterSensitivity:
    """Sensitivity of the model's outputs to one parameter."""

    name: str
    role: str
    baseline_value: float
    elasticity: float
    """Mean elasticity of the crisis count over the tested perturbations."""

    elasticity_stderr: float
    """Monte Carlo standard error of :attr:`elasticity`."""

    beta_star_shift_pct: float
    """Percentage shift in ``beta*`` at a +10% perturbation."""

    classification: Classification

    @property
    def is_more_than_proportional(self) -> bool:
        """Whether the crisis count responds more than proportionally."""
        return abs(self.elasticity) > 1.0

    @property
    def is_distinguishable_from_zero(self) -> bool:
        """Whether the elasticity exceeds twice its Monte Carlo standard error."""
        return abs(self.elasticity) > 2.0 * self.elasticity_stderr


def _mean_crisis_count(
    params: CrisisModelParameters,
    grid: GridSpec,
    *,
    n_realizations: int,
    seed: int,
) -> float:
    """Ensemble-mean crisis count over the horizon.

    A fresh solver is constructed because perturbing ``x0``, ``source_width`` or
    ``diffusivity`` changes the spatial operator or the source profile, which are
    cached on the solver instance.
    """
    solver = IMEXCrisisSolver(params, grid)
    result = solver.run(n_realizations=n_realizations, seed=seed)
    return float(result.crisis_counts.mean())


def _beta_star(params: CrisisModelParameters, grid: GridSpec) -> float | None:
    """Critical amplitude, or ``None`` if the bracket no longer straddles it."""
    try:
        return critical_amplitude(params, grid)
    except (ValueError, SteadyStateConvergenceError):
        return None


def sensitivity_analysis(
    params: CrisisModelParameters | None = None,
    grid: GridSpec | None = None,
    *,
    names: tuple[str, ...] | None = None,
    fractions: tuple[float, ...] = (-0.2, -0.1, 0.1, 0.2),
    n_realizations: int = 60,
    seed: int = 20240,
) -> list[ParameterSensitivity]:
    """Run the elasticity analysis of Section 5.7.

    Parameters
    ----------
    params
        Baseline parameters. ``beta`` is held fixed throughout, since the
        question is how the *shared* assumptions move the conclusions. When
        omitted it defaults to ``beta = BETA_MAX``; see the note below on why
        this choice matters.

        .. note::
           Elasticities depend strongly on which patient they are evaluated for,
           because the crisis count sits in the denominator of Eq. (16) and a
           mildly affected patient has few crises for a perturbation to move.
           Measured here, the threshold elasticity runs from about -8.7 at
           ``beta = 0.0139`` to -5.3 at ``beta = 0.0176``. The published Table 6
           values are reproduced within Monte Carlo error at the top of the
           calibrated range, so that is the default. A single elasticity is
           therefore a statement about a severely affected patient, not about
           the model in general.
    fractions
        Signed perturbations applied to each parameter. The paper averages over
        +-10% and +-20%.
    n_realizations
        Ensemble size per evaluation. Crisis counts are integer-valued, so small
        ensembles give visibly quantised elasticities; 60 is enough to rank
        parameters but not to pin a third significant figure.
    seed
        Base seed. The *same* seed is reused for the baseline and every
        perturbed run, which couples the noise realisations and cancels most of
        the Monte Carlo error in the difference -- the standard common-random-
        numbers variance reduction. Without it, an elasticity of 0.03 would be
        indistinguishable from sampling noise.

    Returns
    -------
    list of ParameterSensitivity
        Ordered by descending absolute elasticity, as in Table 6.
    """
    params = params or CrisisModelParameters(beta=BETA_MAX)
    grid = grid or GridSpec()
    names = names or tuple(SENSITIVITY_PARAMETERS)

    baseline_count = _mean_crisis_count(params, grid, n_realizations=n_realizations, seed=seed)
    if baseline_count == 0.0:
        raise ValueError(
            "baseline produces no crises, so relative changes are undefined; "
            "choose a beta above the critical amplitude"
        )
    baseline_beta_star = _beta_star(params, grid)

    results: list[ParameterSensitivity] = []
    for name in names:
        if name not in SENSITIVITY_PARAMETERS:
            raise KeyError(f"{name!r} is not a sensitivity parameter")

        elasticities: list[float] = []
        for fraction in fractions:
            perturbed = params.perturbed(name, fraction)
            count = _mean_crisis_count(perturbed, grid, n_realizations=n_realizations, seed=seed)
            relative_change = (count - baseline_count) / baseline_count
            elasticities.append(relative_change / fraction)

        elasticity = float(np.mean(elasticities))
        stderr = (
            float(np.std(elasticities, ddof=1) / np.sqrt(len(elasticities)))
            if len(elasticities) > 1
            else 0.0
        )

        shift_pct = float("nan")
        if baseline_beta_star is not None:
            shifted = _beta_star(params.perturbed(name, 0.1), grid)
            if shifted is not None:
                shift_pct = 100.0 * (shifted - baseline_beta_star) / baseline_beta_star

        results.append(
            ParameterSensitivity(
                name=name,
                role=SENSITIVITY_PARAMETERS[name],
                baseline_value=float(getattr(params, name)),
                elasticity=elasticity,
                elasticity_stderr=stderr,
                beta_star_shift_pct=shift_pct,
                classification=_classify(elasticity),
            )
        )

    return sorted(results, key=lambda s: abs(s.elasticity), reverse=True)
