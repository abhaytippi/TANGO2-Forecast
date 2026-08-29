"""Command-line entry point: regenerate the paper's numerical verification.

Running ``tango2-verify`` recomputes, from source, every number in Section 4 and
Eq. (14) of the paper and prints them beside the published values. It exists so
that a reader can check the reproduction in one command without reading any
code, and so that CI has a human-readable artefact when a number moves.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .disclaimer import FORECAST_RESEARCH_USE_ONLY
from .forecast import (
    BETA_CRITICAL_PUBLISHED,
    BETA_MAX,
    BETA_MIN,
    CrisisModelParameters,
    IMEXCrisisSolver,
    critical_amplitude,
    mass_conservation_drift,
    max_amplification,
    sensitivity_analysis,
    spatial_convergence,
    temporal_convergence,
)

_RULE = "-" * 74


def _header(title: str) -> None:
    print(f"\n{title}\n{_RULE}")


def _run_solver_checks() -> None:
    _header("Numerical verification of the IMEX solver (paper, Section 4)")

    print("Spatial refinement, one-sided Neumann rows (the paper's scheme)")
    print(f"  {'N':>6}  {'rel. L2 error':>14}  {'order':>6}   published order")
    published_spatial = [None, 1.05, 1.10, 1.23, 1.59]
    for level, published in zip(spatial_convergence(), published_spatial, strict=True):
        order = "--" if level.order is None else f"{level.order:.2f}"
        ref = "--" if published is None else f"{published:.2f}"
        print(f"  {int(level.resolution):>6}  {level.error:>14.3e}  {order:>6}   {ref:>14}")

    print("\nSpatial refinement, mirrored ghost rows (second-order boundary)")
    print("  Diagnostic: recovers the interior stencil's order, confirming the")
    print("  paper's attribution of first order to the boundary treatment.")
    print(f"  {'N':>6}  {'rel. L2 error':>14}  {'order':>6}")
    for level in spatial_convergence(boundary="mirrored"):
        order = "--" if level.order is None else f"{level.order:.2f}"
        print(f"  {int(level.resolution):>6}  {level.error:>14.3e}  {order:>6}")

    print("\nTemporal refinement")
    print(f"  {'dt (d)':>8}  {'rel. L2 error':>14}  {'order':>6}   published order")
    published_temporal = [None, 1.05, 1.10, 1.22]
    for level, published in zip(temporal_convergence(), published_temporal, strict=True):
        order = "--" if level.order is None else f"{level.order:.2f}"
        ref = "--" if published is None else f"{published:.2f}"
        print(f"  {level.resolution:>8}  {level.error:>14.3e}  {order:>6}   {ref:>14}")

    drift = mass_conservation_drift()
    amplification = max_amplification()
    print(f"\n  Mass drift, pure diffusion   {drift:.2e}   (published 1.4e-12)")
    print(f"  von Neumann max|g|           {amplification:.6f}   (published 1.000000)")


def _run_bifurcation_check() -> None:
    _header("Critical source amplitude (paper, Eq. 14)")
    beta_star = critical_amplitude()
    deviation = 100.0 * (beta_star - BETA_CRITICAL_PUBLISHED) / BETA_CRITICAL_PUBLISHED
    print(f"  computed   beta* = {beta_star:.6f}")
    print(f"  published  beta* = {BETA_CRITICAL_PUBLISHED:.6f}   ({deviation:+.2f}%)")


def _run_crisis_statistics(n_realizations: int) -> None:
    _header(f"Severity strata (paper, Section 5.5), {n_realizations} realisations")
    solver = IMEXCrisisSolver()
    strata = [
        ("high", 0.0167, "5.0 +- 0.1", "229 +- 7"),
        ("moderate", 0.0158, "4.0 +- 0.0", "256 +- 9"),
        ("mild", 0.0139, "2.8 +- 0.4", "347 +- 17"),
    ]
    print(
        f"  {'stratum':<9} {'beta':>8}  {'crises':>13} {'published':>11}"
        f"  {'first crisis':>14} {'published':>11}"
    )
    for name, beta, pub_count, pub_day in strata:
        s = solver.run(beta=beta, n_realizations=n_realizations, seed=2024).summary()
        counts = f"{s['mean_crisis_count']:.1f} +- {s['sd_crisis_count']:.1f}"
        day = f"{s['mean_first_crisis_day']:.0f} +- {s['sd_first_crisis_day']:.0f}"
        print(f"  {name:<9} {beta:>8.4f}  {counts:>13} {pub_count:>11}  {day:>14} {pub_day:>11}")

    sub = solver.run(beta=BETA_MIN, n_realizations=n_realizations, seed=1234).summary()
    deterministic = solver.run(beta=BETA_MIN, stochastic=False, seed=None)
    print(f"\n  Noise-induced transition at beta = {BETA_MIN} (just below beta*)")
    print(f"    deterministic crises           {deterministic.crisis_counts[0]}   (published 0)")
    print(
        f"    P(at least one crisis)         {sub['probability_any_crisis']:.2f}   (published 0.83)"
    )


def _run_sensitivity(n_realizations: int) -> None:
    _header(f"Parameter sensitivity (paper, Table 6), beta = {BETA_MAX}")
    published = {
        "r_star": (-4.94, 28.1),
        "delta": (-2.72, 16.3),
        "source_width": (2.33, -8.1),
        "alpha": (1.66, -9.0),
        "kappa": (0.76, 0.0),
        "diffusivity": (-0.71, 2.4),
        "x0": (0.33, -2.3),
        "noise_intensity": (0.03, 0.0),
    }
    print(
        f"  {'parameter':<16}{'role':<20}{'S':>7}{'+-se':>6}{'pub':>7}"
        f"{'dbeta* %':>10}{'pub':>7}  class"
    )
    for s in sensitivity_analysis(
        CrisisModelParameters(beta=BETA_MAX), n_realizations=n_realizations
    ):
        pub_s, pub_b = published[s.name]
        print(
            f"  {s.name:<16}{s.role:<20}{s.elasticity:>7.2f}{s.elasticity_stderr:>6.2f}"
            f"{pub_s:>7.2f}{s.beta_star_shift_pct:>10.2f}{pub_b:>7.1f}  {s.classification}"
        )
    print("\n  Elasticities are evaluated at the top of the calibrated range. They")
    print("  depend strongly on which patient they describe: the threshold elasticity")
    print("  runs from about -8.7 at beta = 0.0139 to -5.3 at beta = 0.0176.")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``tango2-verify``."""
    parser = argparse.ArgumentParser(
        prog="tango2-verify",
        description="Recompute the paper's numerical verification from source.",
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=120,
        help="stochastic realisations per severity stratum (default: 120)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="also run the Table 6 elasticity analysis (slow: several minutes)",
    )
    parser.add_argument(
        "--skip-stochastic",
        action="store_true",
        help="run only the deterministic solver and bifurcation checks",
    )
    args = parser.parse_args(argv)

    np.set_printoptions(precision=4)
    started = time.perf_counter()

    print("TANGO2-Forecast -- numerical verification")
    print(_RULE)
    print("These checks establish that the equations are solved correctly.")
    print("They say nothing about whether the model is biologically right.")

    _run_solver_checks()
    _run_bifurcation_check()
    if not args.skip_stochastic:
        _run_crisis_statistics(args.realizations)
    if args.sensitivity:
        _run_sensitivity(args.realizations)

    print(f"\n{_RULE}\n{FORECAST_RESEARCH_USE_ONLY}")
    print(f"\nCompleted in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
