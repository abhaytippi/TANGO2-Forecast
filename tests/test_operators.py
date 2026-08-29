"""Operator-level tests: no-flux Laplacian, conservation, tridiagonal solves."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tango2_forecast.forecast import (
    CrisisModelParameters,
    GridSpec,
    amplification_factor,
    build_implicit_operator,
    laplacian_bands,
    thomas_solve,
)


@pytest.fixture(params=["one_sided", "mirrored"])
def grid(request) -> GridSpec:
    return GridSpec(n_nodes=64, boundary=request.param)


def _dense(lower, diag, upper) -> np.ndarray:
    return np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)


def test_laplacian_annihilates_constants(grid: GridSpec) -> None:
    """No-flux diffusion must leave a spatially uniform field untouched."""
    lap = _dense(*laplacian_bands(grid))
    constant = np.full(grid.n_nodes, 0.37)
    assert np.allclose(lap @ constant, 0.0, atol=1e-9)


def test_laplacian_conserves_the_matching_quadrature(grid: GridSpec) -> None:
    """The conserved measure depends on the boundary treatment.

    ``one_sided`` rows annihilate the rectangle sum; ``mirrored`` rows annihilate
    the trapezoidal sum. ``GridSpec.quadrature_weights`` must return whichever
    one the operator actually conserves.
    """
    lap = _dense(*laplacian_bands(grid))
    weights = grid.quadrature_weights
    rng = np.random.default_rng(0)
    for _ in range(5):
        field = rng.uniform(0.0, 1.0, grid.n_nodes)
        assert abs(weights @ (lap @ field)) < 1e-8


def test_wrong_quadrature_would_not_conserve() -> None:
    """Guard against the test above passing for a trivial reason."""
    grid = GridSpec(n_nodes=64, boundary="mirrored")
    lap = _dense(*laplacian_bands(grid))
    rectangle = np.full(grid.n_nodes, grid.dx)
    field = np.exp(-((grid.x - 0.3) ** 2) / 0.01)
    assert abs(rectangle @ (lap @ field)) > 1e-6


def test_laplacian_matches_second_derivative_in_the_interior(grid: GridSpec) -> None:
    """Interior rows are second-order accurate on a smooth test function."""
    fine = GridSpec(n_nodes=2001, boundary=grid.boundary)
    x = fine.x
    field = np.sin(2.0 * np.pi * x)
    exact = -((2.0 * np.pi) ** 2) * field
    lap = _dense(*laplacian_bands(fine))
    interior = slice(5, -5)
    assert np.max(np.abs((lap @ field)[interior] - exact[interior])) < 1e-2


def test_implicit_operator_is_diagonally_dominant(grid: GridSpec) -> None:
    op = build_implicit_operator(grid, diffusivity=0.0012, dt_days=0.5)
    assert op.is_diagonally_dominant()


def test_thomas_agrees_with_lapack(grid: GridSpec) -> None:
    """The reference O(N) Thomas solver and the production LAPACK path agree."""
    op = build_implicit_operator(grid, diffusivity=0.0012, dt_days=0.5)
    rng = np.random.default_rng(7)
    rhs = rng.normal(size=grid.n_nodes)
    reference = thomas_solve(op.lower, op.diag, op.upper, rhs)
    assert np.allclose(reference, op.solve(rhs), rtol=1e-12, atol=1e-14)


def test_solve_inverts_matvec(grid: GridSpec) -> None:
    op = build_implicit_operator(grid, diffusivity=0.0012, dt_days=0.5)
    rng = np.random.default_rng(11)
    x = rng.normal(size=grid.n_nodes)
    assert np.allclose(op.solve(op.matvec(x)), x, rtol=1e-10, atol=1e-12)


def test_batched_solve_matches_single_column(grid: GridSpec) -> None:
    """Ensemble batching must not change any individual solve."""
    op = build_implicit_operator(grid, diffusivity=0.0012, dt_days=0.5)
    rng = np.random.default_rng(13)
    batch = rng.normal(size=(grid.n_nodes, 6))
    batched = op.solve(batch)
    for j in range(batch.shape[1]):
        assert np.allclose(batched[:, j], op.solve(batch[:, j]), rtol=1e-12, atol=1e-14)


def test_amplification_never_exceeds_one() -> None:
    """Proposition 1: the implicit diffusive update amplifies no mode."""
    grid = GridSpec(n_nodes=260)
    theta = np.linspace(-np.pi, np.pi, 2001)
    for dt in (0.0625, 0.5, 5.0, 500.0):
        g = amplification_factor(grid, 0.0012, dt, theta)
        assert np.all(g > 0.0)
        assert np.max(g) <= 1.0 + 1e-15


class TestParameterValidation:
    def test_frozen(self) -> None:
        params = CrisisModelParameters()
        with pytest.raises(ValidationError):
            params.beta = 0.02  # type: ignore[misc]

    def test_rejects_growth_without_clearance(self) -> None:
        with pytest.raises(ValueError, match="clearance"):
            CrisisModelParameters(delta=0.0)

    def test_permits_pure_diffusion(self) -> None:
        CrisisModelParameters(alpha=0.0, delta=0.0)

    def test_rejects_supra_threshold_initial_condition(self) -> None:
        with pytest.raises(ValueError, match="r_initial"):
            CrisisModelParameters(r_initial=0.9, r_star=0.72)

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            CrisisModelParameters(gamma=1.0)  # type: ignore[call-arg]

    def test_source_is_peak_normalised(self) -> None:
        params = CrisisModelParameters()
        x = GridSpec(n_nodes=4001).x
        source = params.source_profile(x)
        assert source.max() == pytest.approx(1.0, abs=1e-6)
        assert x[source.argmax()] == pytest.approx(params.x0, abs=1e-3)

    def test_well_mixed_fixed_point_exceeds_one_over_calibrated_range(self) -> None:
        """Check the well-mixed limit crosses threshold for every calibrated beta.

        Paper, Section 3.7: this is clinically false, and is what motivates the
        spatial formulation.
        """
        for beta in (0.0121, 0.0158, 0.0176):
            assert CrisisModelParameters(beta=beta).well_mixed_fixed_point() > 1.0

    def test_perturbed_scales_the_named_field(self) -> None:
        params = CrisisModelParameters()
        assert params.perturbed("delta", 0.1).delta == pytest.approx(0.0077)
        with pytest.raises(KeyError):
            params.perturbed("not_a_parameter", 0.1)
