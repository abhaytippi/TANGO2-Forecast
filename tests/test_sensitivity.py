"""Regression tests for the parameter sensitivity analysis (paper, Section 5.7).

The two halves of Table 6 are checked very differently, because they have very
different precision.

The shift in ``beta*`` is a deterministic steady-state quantity with no Monte
Carlo error, so it is asserted tightly against the published percentages. The
elasticity of the crisis count is a stochastic estimate whose value also depends
on which patient it is evaluated for, so the assertions there target the sign,
the ordering, and the sensitive/moderate/robust classification -- which is what
the paper's four stated findings actually rest on.

Ensembles are kept small to keep the suite fast; the tolerances below are set
accordingly and are deliberately looser than the figures a full run produces.
"""

from __future__ import annotations

import pytest

from tango2_forecast.forecast import BETA_MAX, CrisisModelParameters, sensitivity_analysis

# Table 6: parameter -> (published elasticity, published shift in beta* at +10%).
PUBLISHED: dict[str, tuple[float, float]] = {
    "r_star": (-4.94, 28.1),
    "delta": (-2.72, 16.3),
    "source_width": (2.33, -8.1),
    "alpha": (1.66, -9.0),
    "kappa": (0.76, 0.0),
    "diffusivity": (-0.71, 2.4),
    "x0": (0.33, -2.3),
    "noise_intensity": (0.03, 0.0),
}


@pytest.fixture(scope="module")
def analysis():
    """Sensitivity results at the top of the calibrated cohort range."""
    return {
        s.name: s
        for s in sensitivity_analysis(CrisisModelParameters(beta=BETA_MAX), n_realizations=24)
    }


class TestCriticalAmplitudeShifts:
    """The deterministic half of Table 6, asserted tightly."""

    @pytest.mark.parametrize("name", list(PUBLISHED))
    def test_matches_published_shift(self, analysis, name: str) -> None:
        published = PUBLISHED[name][1]
        assert analysis[name].beta_star_shift_pct == pytest.approx(published, abs=0.2)

    def test_reset_leaves_beta_star_exactly_unchanged(self, analysis) -> None:
        """Check the reset cannot move a steady state.

        Section 5.7: a reset acts only after a threshold crossing, so any
        nonzero shift here would indicate a bug rather than a finding.
        """
        assert analysis["kappa"].beta_star_shift_pct == 0.0

    def test_noise_leaves_beta_star_exactly_unchanged(self, analysis) -> None:
        """The noise intensity does not appear in the steady-state problem."""
        assert analysis["noise_intensity"].beta_star_shift_pct == 0.0

    def test_threshold_dominates_the_critical_amplitude(self, analysis) -> None:
        shifts = {n: abs(s.beta_star_shift_pct) for n, s in analysis.items()}
        assert max(shifts, key=shifts.__getitem__) == "r_star"


class TestElasticities:
    """The stochastic half: sign, ordering, and classification."""

    @pytest.mark.parametrize("name", list(PUBLISHED))
    def test_sign_matches(self, analysis, name: str) -> None:
        published = PUBLISHED[name][0]
        observed = analysis[name].elasticity
        if abs(published) < 0.1:  # sigma_eta is indistinguishable from zero
            assert abs(observed) < 0.5
        else:
            assert observed * published > 0.0, f"{name}: {observed} vs {published}"

    def test_threshold_and_clearance_dominate(self, analysis) -> None:
        """Check that the threshold and clearance rate dominate.

        Section 5.7, first finding. This is expected rather than alarming: they
        encode the core clinical assumptions, so they *should* dominate.
        """
        ranked = sorted(analysis, key=lambda n: abs(analysis[n].elasticity), reverse=True)
        assert set(ranked[:2]) == {"r_star", "delta"}

    def test_locus_and_noise_are_robust(self, analysis) -> None:
        """Check the model is nearly insensitive to the locus and the noise.

        Second finding, and the answer to the obvious objection that the choice
        x0 = 0.58 is arbitrary.
        """
        assert abs(analysis["x0"].elasticity) < 1.0
        assert abs(analysis["noise_intensity"].elasticity) < 1.0

    def test_sensitive_parameters_are_classified_as_such(self, analysis) -> None:
        for name in ("r_star", "delta", "source_width", "alpha"):
            assert analysis[name].classification == "sensitive"
            assert analysis[name].is_more_than_proportional

    def test_noise_elasticity_is_not_distinguishable_from_zero(self, analysis) -> None:
        assert not analysis["noise_intensity"].is_distinguishable_from_zero


class TestApiContract:
    def test_rejects_unknown_parameter(self) -> None:
        with pytest.raises(KeyError):
            sensitivity_analysis(names=("not_a_parameter",), n_realizations=2)

    def test_rejects_a_crisis_free_baseline(self) -> None:
        """Relative change is undefined when the baseline produces no crises."""
        params = CrisisModelParameters(beta=0.001, noise_intensity=0.0)
        with pytest.raises(ValueError, match="no crises"):
            sensitivity_analysis(params, names=("alpha",), n_realizations=2)

    def test_results_are_ordered_by_absolute_elasticity(self, analysis) -> None:
        results = sensitivity_analysis(
            CrisisModelParameters(beta=BETA_MAX),
            names=("r_star", "x0", "delta"),
            n_realizations=8,
        )
        magnitudes = [abs(s.elasticity) for s in results]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_reports_a_monte_carlo_standard_error(self, analysis) -> None:
        assert all(s.elasticity_stderr >= 0.0 for s in analysis.values())
