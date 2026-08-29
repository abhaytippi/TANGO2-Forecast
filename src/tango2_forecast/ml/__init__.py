"""Machine learning diagnostic engine (Module B)."""

from .classifier import (
    Contribution,
    DiagnosticModel,
    EvaluationReport,
    PatientPrediction,
    train_model,
)
from .cohort import (
    PATHOGNOMONIC_MARKERS,
    PHENOTYPE_CATEGORIES,
    PHENOTYPES,
    Phenotype,
    SyntheticCohort,
    generate_synthetic_cohort,
)

__all__ = [
    "PATHOGNOMONIC_MARKERS",
    "PHENOTYPES",
    "PHENOTYPE_CATEGORIES",
    "Contribution",
    "DiagnosticModel",
    "EvaluationReport",
    "PatientPrediction",
    "Phenotype",
    "SyntheticCohort",
    "generate_synthetic_cohort",
    "train_model",
]
