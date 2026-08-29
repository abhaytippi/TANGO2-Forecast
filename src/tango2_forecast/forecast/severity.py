"""Mapping baseline phenotype burden to the patient-specific source amplitude.

Implements Eq. (8) of the paper:

``sigma_i = (n_i - n_min) / (n_max - n_min)``,
``beta_i = beta_min + (beta_max - beta_min) sigma_i``

where ``n_i`` is the count of retained baseline HPO terms for patient ``i``.

Two things about this mapping deserve to be stated plainly, because they bound
every downstream forecast:

1. The normalisation is **cohort-relative**. ``n_min`` and ``n_max`` come from
   the reference cohort, so ``sigma_i`` expresses a patient's burden relative to
   that cohort and not on any absolute scale. Applying the mapping to a patient
   phenotyped less deeply than the reference cohort will understate severity.
2. ``beta`` was never fitted to observed crisis dates. The range
   ``[0.0121, 0.0176]`` was chosen so that the cohort straddles ``beta*``
   (paper, Section 3.5). It reproduces a qualitative clinical observation; it is
   not a measured quantity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .parameters import BETA_MAX, BETA_MIN

__all__ = ["RegimeAssessment", "SeverityScale", "classify_regime"]

Regime = Literal["quiescent", "recurrent"]


@dataclass(frozen=True, slots=True)
class SeverityScale:
    """Cohort-relative min--max normalisation of baseline term counts.

    Fit once on a reference cohort, then applied to individual patients. Storing
    the fitted bounds (rather than recomputing them per call) is what makes a
    severity score comparable between two runs of the software.
    """

    n_min: int
    n_max: int
    beta_min: float = BETA_MIN
    beta_max: float = BETA_MAX

    def __post_init__(self) -> None:
        """Validate that the fitted bounds define a usable normalisation."""
        if self.n_max <= self.n_min:
            raise ValueError("n_max must exceed n_min for the normalisation to be defined")
        if self.beta_max <= self.beta_min:
            raise ValueError("beta_max must exceed beta_min")

    @classmethod
    def fit(
        cls,
        term_counts: Sequence[int] | np.ndarray,
        *,
        beta_min: float = BETA_MIN,
        beta_max: float = BETA_MAX,
    ) -> SeverityScale:
        """Fit the scale to a reference cohort's baseline term counts."""
        counts = np.asarray(term_counts, dtype=int)
        if counts.size == 0:
            raise ValueError("cannot fit a severity scale to an empty cohort")
        return cls(int(counts.min()), int(counts.max()), beta_min, beta_max)

    # ------------------------------------------------------------------
    def severity(self, term_count: int | np.ndarray) -> np.ndarray | float:
        """Normalised severity ``sigma`` in ``[0, 1]``.

        Counts outside the fitted range are clipped rather than extrapolated,
        because ``beta`` outside the calibrated interval has no support in the
        paper and the model's behaviour there is not characterised.
        """
        raw = (np.asarray(term_count, dtype=float) - self.n_min) / (self.n_max - self.n_min)
        clipped = np.clip(raw, 0.0, 1.0)
        return float(clipped) if np.isscalar(term_count) else clipped

    def beta(self, term_count: int | np.ndarray) -> np.ndarray | float:
        """Source amplitude ``beta_i`` for a patient with ``term_count`` terms."""
        sigma = self.severity(term_count)
        return self.beta_min + (self.beta_max - self.beta_min) * sigma

    def is_extrapolating(self, term_count: int) -> bool:
        """Whether the count falls outside the fitted cohort range."""
        return not (self.n_min <= term_count <= self.n_max)


@dataclass(frozen=True, slots=True)
class RegimeAssessment:
    """Where a patient sits relative to the critical amplitude ``beta*``."""

    beta: float
    beta_critical: float
    regime: Regime

    @property
    def margin(self) -> float:
        """``beta - beta*``. Negative means nominally quiescent."""
        return self.beta - self.beta_critical

    @property
    def relative_margin(self) -> float:
        """Margin as a fraction of ``beta*``."""
        return self.margin / self.beta_critical

    def clinical_note(self) -> str:
        """Plain-language interpretation, phrased to avoid false reassurance.

        The wording follows the paper's Section 6.2 precisely: a patient below
        ``beta*`` is *less likely* to decompensate, not safe. In the model, 83%
        of stochastic realisations just below ``beta*`` still contain at least
        one crisis, so a deterministic reading would wrongly assign zero risk.
        """
        if self.regime == "recurrent":
            return (
                "Model places this patient above the critical source amplitude, in the "
                "regime where crises recur endogenously. In the framework's terms this "
                "would argue for closer cardiac surveillance, a lower admission threshold "
                "during febrile illness, and a more aggressive sick-day plan."
            )
        return (
            "Model places this patient below the critical source amplitude, in the "
            "nominally quiescent regime. This does not mean zero risk: stochastic "
            "triggers alone produce crises in a large fraction of simulated "
            "realisations just below the critical value. The correct reading is lower "
            "probability, not safety."
        )


def classify_regime(beta: float, beta_critical: float) -> RegimeAssessment:
    """Classify a patient's source amplitude relative to ``beta*``."""
    regime: Regime = "recurrent" if beta >= beta_critical else "quiescent"
    return RegimeAssessment(beta=beta, beta_critical=beta_critical, regime=regime)
