"""Minimal end to end example: assess a patient, forecast, and write a report.

Run with:  python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tango2_forecast.forecast import (
    IMEXCrisisSolver,
    SeverityScale,
    classify_regime,
    critical_amplitude,
)
from tango2_forecast.ml import PHENOTYPES, train_model
from tango2_forecast.reports import ReportData, build_pdf

# 1. Fit the diagnostic model. Without a cohort argument this uses the
#    synthetic generator and labels itself accordingly.
model = train_model()
print(f"Model provenance: {model.provenance.upper()}")
print(f"Cross validated AUC: {model.evaluation.auc:.3f}")

# 2. Assess one patient from the baseline phenotypes they present with.
observed = [
    "Dysarthria",
    "Motor delay",
    "Gait disturbance",
    "Intellectual disability",
    "Poor speech",
    "Muscle weakness",
]
prediction = model.predict(observed, pathognomonic=["Paroxysmal lethargy"])
print(f"\nProbability: {prediction.probability:.1%}  ({prediction.confidence_band})")
print(prediction.plain_language())

# 3. Place the patient on the crisis model's bifurcation.
scale = SeverityScale.fit([1, len(PHENOTYPES)])
beta = float(scale.beta(len(observed)))
beta_star = critical_amplitude()
assessment = classify_regime(beta, beta_star)
print(f"\nSource amplitude {beta:.5f} vs critical {beta_star:.5f}")
print(f"Regime: {assessment.regime.upper()}")

# 4. Run an ensemble forecast. Research use only.
result = IMEXCrisisSolver().run(beta=beta, n_realizations=50, seed=0)
summary = result.summary()
print(f"Mean simulated crises over 900 days: {summary['mean_crisis_count']:.2f}")

# 5. Write a PDF report.
report = ReportData(
    patient_reference="EXAMPLE-001",
    clinician="Example User",
    phenotypes=observed,
    pathognomonic_markers=["Paroxysmal lethargy"],
    probability=prediction.probability,
    predicted_label=prediction.predicted_label,
    confidence_band=prediction.confidence_band,
    interpretation=prediction.plain_language(),
    top_contributors=[(c.phenotype, c.weight) for c in prediction.top_contributors],
    model_provenance=model.provenance,
    model_version=model.model_version,
    auc=model.evaluation.auc,
    sensitivity=model.evaluation.sensitivity,
    specificity=model.evaluation.specificity,
    brier=model.evaluation.brier,
    include_forecast=True,
    beta=beta,
    beta_critical=beta_star,
    regime=assessment.regime,
    regime_note=assessment.clinical_note(),
    mean_crisis_count=summary["mean_crisis_count"],
    first_crisis_day=summary["mean_first_crisis_day"],
    probability_any_crisis=summary["probability_any_crisis"],
    n_realizations=result.n_realizations,
    seed=0,
)
output = Path(__file__).parent / "example_report.pdf"
output.write_bytes(build_pdf(report))
print(f"\nReport written to {output}")
