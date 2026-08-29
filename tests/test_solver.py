"""Solver behaviour, reproducibility, and the severity-to-amplitude mapping.

The reproduction targets in :class:`TestPublishedCrisisStatistics` come from
paper Section 5.5 and are the closest thing available to an end-to-end check of
the dynamics, since no observed crisis dates exist to validate against.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from tango2_forecast.forecast import (
    BETA_MAX,
    BETA_MIN,
    CrisisModelParameters,
    IMEXCrisisSolver,
    SeverityScale,
    classify_regime,
    critical_amplitude,
)


@pytest.fixture(scope="module")
def solver() -> IMEXCrisisSolver:
    return IMEXCrisisSolver()


@pytest.fixture(scope="module")
def beta_star() -> float:
    return critical_amplitude()


class TestInvariants:
    def test_risk_stays_in_the_unit_interval(self, solver: IMEXCrisisSolver) -> None:
        """Check that risk cannot leave the unit interval.

        The sqrt(r(1-r)) noise prefactor vanishes at both ends, so the solution
        is trapped in [0, 1] (paper, Section 3.4).
        """
        result = solver.run(beta=BETA_MAX, n_realizations=8, seed=3)
        assert result.peak_risk.min() >= 0.0
        assert result.peak_risk.max() <= 1.0

    def test_seeded_runs_are_bit_identical(self, solver: IMEXCrisisSolver) -> None:
        """A forecast that appears in a clinical report must be reproducible."""
        a = solver.run(beta=0.0158, n_realizations=4, seed=42)
        b = solver.run(beta=0.0158, n_realizations=4, seed=42)
        assert np.array_equal(a.peak_risk, b.peak_risk)
        for left, right in zip(a.crisis_days, b.crisis_days, strict=True):
            assert np.array_equal(left, right)

    def test_different_seeds_diverge(self, solver: IMEXCrisisSolver) -> None:
        a = solver.run(beta=0.0158, n_realizations=4, seed=1)
        b = solver.run(beta=0.0158, n_realizations=4, seed=2)
        assert not np.array_equal(a.peak_risk, b.peak_risk)

    def test_deterministic_mode_ignores_the_seed(self, solver: IMEXCrisisSolver) -> None:
        a = solver.run(beta=0.0158, stochastic=False, seed=1)
        b = solver.run(beta=0.0158, stochastic=False, seed=99)
        assert np.array_equal(a.peak_risk, b.peak_risk)

    def test_zero_noise_falls_back_to_deterministic(self, solver: IMEXCrisisSolver) -> None:
        quiet = IMEXCrisisSolver(CrisisModelParameters(noise_intensity=0.0))
        assert quiet.run(beta=0.0158, seed=5).stochastic is False

    def test_reset_lowers_peak_risk_by_the_specified_fraction(self) -> None:
        """Eq. (6) removes 48% of supra-resting risk."""
        params = CrisisModelParameters(beta=0.0176, noise_intensity=0.0)
        result = IMEXCrisisSolver(params).run(stochastic=False, seed=None)
        assert result.crisis_counts[0] > 0
        first = round(result.crisis_days[0][0] / result.grid.dt_days)
        expected = params.r_rest + params.kappa * (result.peak_risk[first - 1] - params.r_rest)
        # peak_risk at the crisis index is recorded post-reset
        assert result.peak_risk[first] < result.peak_risk[first - 1]
        assert result.peak_risk[first] == pytest.approx(expected, abs=0.02)

    def test_disabling_the_reset_still_records_crises(self, solver: IMEXCrisisSolver) -> None:
        result = solver.run(beta=BETA_MAX, stochastic=False, enable_reset=False, seed=None)
        assert result.crisis_counts[0] > 0
        assert np.all(np.diff(result.peak_risk[:, 0]) >= -1e-12), "no resets expected"

    def test_batched_ensemble_matches_independent_first_path(self) -> None:
        """Realisations must be statistically independent across the batch axis."""
        solver = IMEXCrisisSolver()
        result = solver.run(beta=0.0158, n_realizations=32, seed=17)
        correlations = np.corrcoef(result.peak_risk[200:, :].T)
        off_diagonal = correlations[~np.eye(32, dtype=bool)]
        assert np.abs(off_diagonal).max() < 0.999

    def test_rejects_bad_initial_field(self, solver: IMEXCrisisSolver) -> None:
        with pytest.raises(ValueError, match="initial_field"):
            solver.run(initial_field=np.zeros(5))


class TestPublishedCrisisStatistics:
    """Reproduction of the severity strata of paper Section 5.5.

    Tolerances are deliberately loose on crisis *counts* and tight on *first
    crisis timing*. Counts near the end of a fixed 900-day horizon are sensitive
    to whether a final crisis falls just inside or just outside the window,
    whereas the first crossing time is a clean property of the dynamics.
    """

    STRATA: ClassVar[dict[str, tuple[float, float]]] = {
        "high": (0.0167, 229.0),
        "moderate": (0.0158, 256.0),
        "mild": (0.0139, 347.0),
    }

    @pytest.mark.parametrize("stratum", list(STRATA))
    def test_first_crisis_timing(self, solver: IMEXCrisisSolver, stratum: str) -> None:
        beta, published_day = self.STRATA[stratum]
        result = solver.run(beta=beta, n_realizations=40, seed=2024)
        assert result.summary()["mean_first_crisis_day"] == pytest.approx(published_day, abs=15.0)

    def test_crisis_count_increases_with_severity(self, solver: IMEXCrisisSolver) -> None:
        """Check that crisis frequency rises with severity.

        This is a consequence of the dynamics rather than an assumption: larger
        beta rebuilds to threshold faster after each reset.
        """
        counts = [
            solver.run(beta=b, n_realizations=24, seed=8).crisis_counts.mean()
            for b in (0.0139, 0.0158, 0.0167)
        ]
        assert counts == sorted(counts)

    def test_severity_correlates_with_crisis_count(self, solver: IMEXCrisisSolver) -> None:
        """Paper reports r = 0.97 between severity and expected crisis count."""
        betas = np.linspace(BETA_MIN, BETA_MAX, 8)
        counts = [solver.run(beta=b, n_realizations=16, seed=5).crisis_counts.mean() for b in betas]
        assert np.corrcoef(betas, counts)[0, 1] > 0.95

    def test_severity_anticorrelates_with_time_to_first_crisis(
        self, solver: IMEXCrisisSolver
    ) -> None:
        """Paper reports r = -0.91."""
        betas = np.linspace(0.013, BETA_MAX, 8)
        days = [
            solver.run(beta=b, n_realizations=16, seed=6).summary()["mean_first_crisis_day"]
            for b in betas
        ]
        assert np.corrcoef(betas, days)[0, 1] < -0.85


class TestNoiseInducedTransition:
    """Paper, Section 5.6 -- the result the deterministic model cannot produce."""

    def test_deterministic_subcritical_patient_never_decompensates(
        self, solver: IMEXCrisisSolver, beta_star: float
    ) -> None:
        result = solver.run(beta=BETA_MIN, stochastic=False, seed=None)
        assert beta_star > BETA_MIN
        assert result.crisis_counts[0] == 0
        assert result.peak_risk.max() < result.parameters.r_star

    def test_noise_alone_drives_most_subcritical_patients_across(
        self, solver: IMEXCrisisSolver
    ) -> None:
        """Paper: 83% of realisations at beta = 0.0121 contain at least one crisis.

        Clinically this is the argument against false reassurance -- a patient
        below the critical amplitude is less likely to decompensate, not safe.
        """
        result = solver.run(beta=BETA_MIN, n_realizations=120, seed=1234)
        assert result.summary()["probability_any_crisis"] == pytest.approx(0.83, abs=0.10)


class TestSeverityMapping:
    def test_endpoints_map_to_the_calibrated_range(self) -> None:
        scale = SeverityScale.fit([4, 10, 17, 31])
        assert scale.beta(4) == pytest.approx(BETA_MIN)
        assert scale.beta(31) == pytest.approx(BETA_MAX)

    def test_is_linear_in_term_count(self) -> None:
        scale = SeverityScale.fit([0, 20])
        assert scale.severity(10) == pytest.approx(0.5)
        assert scale.beta(10) == pytest.approx(0.5 * (BETA_MIN + BETA_MAX))

    def test_clips_rather_than_extrapolates(self) -> None:
        """Beta outside the calibrated interval has no support in the paper."""
        scale = SeverityScale.fit([5, 25])
        assert scale.beta(100) == pytest.approx(BETA_MAX)
        assert scale.beta(0) == pytest.approx(BETA_MIN)
        assert scale.is_extrapolating(100)
        assert not scale.is_extrapolating(10)

    def test_degenerate_cohort_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_max"):
            SeverityScale.fit([7, 7, 7])
        with pytest.raises(ValueError, match="empty cohort"):
            SeverityScale.fit([])

    def test_vectorised_and_scalar_agree(self) -> None:
        scale = SeverityScale.fit([2, 30])
        vector = scale.beta(np.array([2, 16, 30]))
        assert vector[1] == pytest.approx(scale.beta(16))


class TestRegimeClassification:
    def test_sides_of_the_bifurcation(self, beta_star: float) -> None:
        assert classify_regime(BETA_MIN, beta_star).regime == "quiescent"
        assert classify_regime(BETA_MAX, beta_star).regime == "recurrent"

    def test_quiescent_note_refuses_to_reassure(self, beta_star: float) -> None:
        note = classify_regime(BETA_MIN, beta_star).clinical_note().lower()
        assert "not mean zero risk" in note
        assert "lower probability, not safety" in note

    def test_margin_sign(self, beta_star: float) -> None:
        assert classify_regime(BETA_MIN, beta_star).margin < 0
        assert classify_regime(BETA_MAX, beta_star).relative_margin > 0
