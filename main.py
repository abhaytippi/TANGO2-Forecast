"""TANGO2-Forecast -- run everything.

RUN THIS FILE. In PyCharm, VS Code, or any IDE: open the project folder, open
this file, and press Run. No installation or command line needed -- this script
puts ``src/`` on the import path itself.

What it does, in order:

1. Checks that the required packages are installed, and tells you exactly what
   to type if any are missing.
2. Recomputes every number in the paper's Section 4 (convergence, conservation,
   stability) and Eq. (14) (the critical amplitude) from source, and prints them
   beside the published values.
3. Reproduces the three severity strata of Section 5.5 and the noise-induced
   transition of Section 5.6.
4. Optionally reproduces the Table 6 sensitivity analysis (slower).
5. Runs a worked example: one patient, from baseline phenotype count through to
   a forecast summary.

To skip the slow parts, edit the two switches directly below.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Switches. Edit these.
# --------------------------------------------------------------------------

RUN_SENSITIVITY_ANALYSIS = False
"""Reproduce paper Table 6. Adds roughly one minute."""

REALIZATIONS = 120
"""Stochastic paths per severity stratum. Lower this to run faster."""


# --------------------------------------------------------------------------
# Make the package importable without installing it.
# --------------------------------------------------------------------------

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _check_dependencies() -> None:
    """Fail early with an actionable message rather than a bare ImportError."""
    required = {"numpy": "numpy", "scipy": "scipy", "pydantic": "pydantic"}
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing required packages:", ", ".join(missing))
        print("\nInstall them by running this in the terminal:\n")
        print(f"    {Path(sys.executable).name} -m pip install {' '.join(missing)}\n")
        print("In PyCharm the terminal is at the bottom of the window, or use")
        print("Settings -> Project -> Python Interpreter -> '+' to add packages.")
        raise SystemExit(1)

    # Deliberately checked at runtime rather than left to a syntax error: this
    # script must give a readable message to someone whose IDE picked an older
    # interpreter. Linters target 3.12 and flag the block as dead; it is not.
    if sys.version_info < (3, 12):  # noqa: UP036
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"This project needs Python 3.12 or newer; this is Python {version}.")
        print("In PyCharm: Settings -> Project -> Python Interpreter.")
        raise SystemExit(1)


def _worked_example() -> None:
    """One patient, end to end, as a clinician-facing summary would run it."""
    from tango2_forecast.forecast import (
        IMEXCrisisSolver,
        SeverityScale,
        classify_regime,
        critical_amplitude,
    )

    print("\n" + "=" * 74)
    print("Worked example: one patient, baseline phenotype count to forecast")
    print("=" * 74)

    # Baseline HPO term counts for a reference cohort. Replace with real counts.
    reference_cohort_term_counts = [8, 11, 13, 14, 16, 17, 17, 19, 21, 24, 28, 31]
    scale = SeverityScale.fit(reference_cohort_term_counts)

    patient_term_count = 24
    severity = scale.severity(patient_term_count)
    beta = scale.beta(patient_term_count)
    beta_star = critical_amplitude()
    assessment = classify_regime(beta, beta_star)

    print(f"\n  Baseline retained HPO terms      {patient_term_count}")
    print(f"  Cohort-relative severity sigma   {severity:.3f}")
    print(f"  Source amplitude beta            {beta:.5f}")
    print(f"  Critical amplitude beta*         {beta_star:.5f}")
    print(f"  Margin above/below beta*         {assessment.relative_margin:+.1%}")
    print(f"  Regime                           {assessment.regime.upper()}")

    print("\n  Interpretation")
    for line in _wrap(assessment.clinical_note(), 68):
        print(f"    {line}")

    result = IMEXCrisisSolver().run(beta=beta, n_realizations=REALIZATIONS, seed=0)
    summary = result.summary()

    print(f"\n  Ensemble forecast over 900 days ({REALIZATIONS} realisations, seed 0)")
    print(
        f"    Mean simulated crises          {summary['mean_crisis_count']:.2f}"
        f" +- {summary['sd_crisis_count']:.2f}"
    )
    print(
        f"    Mean day of first crossing     {summary['mean_first_crisis_day']:.0f}"
        f" +- {summary['sd_first_crisis_day']:.0f}"
    )
    print(f"    Realisations with >=1 crisis   {summary['probability_any_crisis']:.0%}")

    print("\n  These are model outputs, not validated predictions. The spread")
    print("  across realisations is the point; no single day is meaningful.")


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to a width without pulling in a dependency."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> int:
    """Run the full demonstration."""
    _check_dependencies()

    from tango2_forecast.cli import main as verification_main

    argv = ["--realizations", str(REALIZATIONS)]
    if RUN_SENSITIVITY_ANALYSIS:
        argv.append("--sensitivity")

    verification_main(argv)
    _worked_example()

    print("\n" + "=" * 74)
    print("Done. Next steps:")
    print("  - Set RUN_SENSITIVITY_ANALYSIS = True at the top of this file to")
    print("    reproduce the paper's Table 6.")
    print("  - Run the test suite: right-click the 'tests' folder -> Run pytest,")
    print("    or type  pytest  in the terminal. 90 tests, about 45 seconds.")
    print("  - Read README.md for the full reproduction table.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
