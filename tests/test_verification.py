"""Regression tests against the paper's published numerical results.

Covers the solver verification of Section 4 and the critical source amplitude of
Eq. (14).

These are the tests that would catch a silent change in the physics rather than
in the plumbing, so they assert against published numbers wherever the paper
provides one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tango2_forecast.forecast import (
    BETA_CRITICAL_PUBLISHED,
    CrisisModelParameters,
    GridSpec,
    bifurcation_curve,
    critical_amplitude,
    mass_conservation_drift,
    max_amplification,
    spatial_convergence,
    steady_state,
    temporal_convergence,
)


class TestSolverVerification:
    def test_mass_conserved_under_pure_diffusion(self) -> None:
        """Paper reports a relative drift of 1.4e-12 after 2000 steps.

        A sign error in the boundary rows shows up here immediately as mass loss.
        """
        for boundary in ("one_sided", "mirrored"):
            drift = mass_conservation_drift(boundary=boundary)
            assert drift < 1e-10, f"{boundary}: drift {drift:.2e} is above round-off"

    def test_amplification_factor_is_exactly_one(self) -> None:
        """Paper: max|g| = 1.000000, matching Proposition 1."""
        assert max_amplification() == pytest.approx(1.0, abs=1e-12)

    def test_spatial_convergence_is_first_order_with_the_paper_scheme(self) -> None:
        """The one-sided Neumann rows dominate, giving order ~1 (paper, Table 2)."""
        levels = spatial_convergence()
        errors = [lv.error for lv in levels]
        assert errors == sorted(errors, reverse=True), "error must fall monotonically"
        observed = [lv.order for lv in levels if lv.order is not None]
        assert all(0.9 < o < 1.8 for o in observed), observed

    def test_mirrored_boundary_recovers_second_order(self) -> None:
        """Check that a second-order boundary recovers second-order convergence.

        Diagnostic for the claim above. If this test and the previous one ever
        agree, the attribution of the first order to the boundary rows is wrong.
        """
        observed = [lv.order for lv in spatial_convergence(boundary="mirrored") if lv.order]
        assert all(1.9 < o < 2.1 for o in observed), observed

    def test_temporal_convergence_is_first_order(self) -> None:
        """IMEX splitting is first order in time (paper, Table 2, right)."""
        levels = temporal_convergence()
        errors = [lv.error for lv in levels]
        assert errors == sorted(errors, reverse=True)
        observed = [lv.order for lv in levels if lv.order is not None]
        assert all(0.9 < o < 1.5 for o in observed), observed


class TestCriticalAmplitude:
    def test_matches_published_value(self) -> None:
        """Paper, Eq. (14): beta* = 0.01273.

        This implementation obtains 0.01270 on the paper's grid. The 0.3%
        difference is far inside the paper's own sensitivity envelope (a 10%
        change in the clearance rate moves beta* by 16%), but the tolerance is
        kept tight enough that a genuine regression would fail.
        """
        beta_star = critical_amplitude()
        assert beta_star == pytest.approx(BETA_CRITICAL_PUBLISHED, rel=0.01)

    def test_is_grid_converged(self) -> None:
        """beta* must not depend materially on the discretisation."""
        values = [
            critical_amplitude(grid=GridSpec(n_nodes=n, boundary="mirrored"))
            for n in (130, 260, 520)
        ]
        # Spread across a 4x refinement is ~1e-6 absolute, i.e. under 0.01% of
        # beta* -- three orders below the sensitivity the paper reports for any
        # physical parameter.
        assert (max(values) - min(values)) / np.mean(values) < 1e-3

    def test_steady_state_peak_equals_threshold_at_beta_star(self) -> None:
        params = CrisisModelParameters()
        beta_star = critical_amplitude(params)
        assert steady_state(params, beta=beta_star).peak == pytest.approx(params.r_star, abs=1e-4)

    def test_reset_cannot_move_the_steady_state(self) -> None:
        """Paper, Section 5.7: kappa leaves beta* *exactly* unchanged.

        This is a correctness check rather than a finding -- the reset acts only
        after a threshold crossing and so cannot influence a steady state, so any
        nonzero shift would indicate a bug.
        """
        baseline = critical_amplitude(CrisisModelParameters(kappa=0.52))
        for kappa in (0.1, 0.3, 0.9):
            assert critical_amplitude(CrisisModelParameters(kappa=kappa)) == baseline

    def test_noise_intensity_cannot_move_the_steady_state(self) -> None:
        """Section 5.7 likewise reports exactly zero shift in beta* for sigma_eta."""
        baseline = critical_amplitude(CrisisModelParameters(noise_intensity=0.045))
        assert critical_amplitude(CrisisModelParameters(noise_intensity=0.2)) == baseline

    def test_critical_value_lies_inside_the_calibrated_cohort_range(self) -> None:
        """Check the bifurcation falls inside the calibrated cohort range.

        This is what makes the mildest patients sit just below it, which is the
        model's account of why some go years without decompensating (Sec. 3.5).
        """
        from tango2_forecast.forecast import BETA_MAX, BETA_MIN

        assert BETA_MIN < critical_amplitude() < BETA_MAX

    def test_bracket_errors_are_explicit(self) -> None:
        with pytest.raises(ValueError, match="lower bracket"):
            critical_amplitude(lower=0.03, upper=0.05)


class TestBifurcationCurve:
    def test_peak_risk_increases_monotonically_with_beta(self) -> None:
        """Paper, Figure 8a: no chaotic regime, so forecasts degrade gracefully."""
        _, peaks = bifurcation_curve(beta_values=np.linspace(0.008, 0.020, 25))
        assert np.all(np.diff(peaks) > 0)

    def test_curve_crosses_the_threshold_at_beta_star(self) -> None:
        params = CrisisModelParameters()
        betas = np.linspace(0.008, 0.020, 121)
        _, peaks = bifurcation_curve(params, beta_values=betas)
        crossing = betas[np.searchsorted(peaks, params.r_star)]
        assert crossing == pytest.approx(critical_amplitude(params), abs=1e-4)
