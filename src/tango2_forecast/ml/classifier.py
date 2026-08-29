"""Random Forest diagnostic classifier (Module B).

Implements the classifier of the paper's Section 3.2: 500 trees on the binary
shared-vocabulary phenotype vector, class weights inversely proportional to
class frequency, evaluated by repeated stratified 10-fold cross-validation with
calibration reported alongside discrimination.

Provenance
----------
Every fitted model carries a :attr:`DiagnosticModel.provenance` string recording
what it was fitted to. A model fitted to the synthetic cohort is labelled
``"synthetic"`` and every surface that displays its output is required to say so.
This is not decoration: a plausible-looking AUC from simulated data is the single
most misleading artefact this software could produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from .cohort import PHENOTYPES, SyntheticCohort, generate_synthetic_cohort

__all__ = [
    "Contribution",
    "DiagnosticModel",
    "EvaluationReport",
    "PatientPrediction",
    "train_model",
]

Provenance = Literal["synthetic", "institutional"]


@dataclass(frozen=True, slots=True)
class Contribution:
    """One phenotype's share of the basis for a single assessment.

    ``weight`` is the phenotype's model importance as a percentage of the
    combined importance of every phenotype the patient presents with, so the
    set of contributions for one patient sums to 100. This is deliberately
    different from the raw Gini importance shown in Model Insights: a raw
    importance of 0.11 has no natural unit a clinician can act on, whereas
    "accounts for 24 percent of this assessment's basis" does, and the two
    numbers answer different questions. Model Insights describes the model in
    general; this describes one patient's assessment.
    """

    phenotype: str
    weight: float


@dataclass(frozen=True, slots=True)
class PatientPrediction:
    """A single patient's prediction, phrased for a clinical reader."""

    probability: float
    predicted_label: str
    confidence_band: str
    top_contributors: list[Contribution]
    """Present phenotypes ranked by their share of the assessment's basis."""

    pathognomonic_present: list[str]
    term_count: int
    provenance: Provenance

    @property
    def is_confident(self) -> bool:
        """Whether the probability sits outside the ambiguous middle."""
        return self.probability < 0.2 or self.probability > 0.8

    def plain_language(self) -> str:
        """One sentence a clinician can read without knowing what an AUC is."""
        percent = f"{self.probability:.0%}"
        if self.probability >= 0.8:
            return (
                f"The baseline phenotype pattern is strongly consistent with TANGO2 "
                f"deficiency ({percent} model probability). This supports referral for "
                f"TANGO2 sequencing."
            )
        if self.probability >= 0.5:
            return (
                f"The pattern is moderately consistent with TANGO2 deficiency "
                f"({percent}). Genetic evaluation with TANGO2 in the differential "
                f"would be reasonable."
            )
        if self.probability >= 0.2:
            return (
                f"The pattern is not typical of TANGO2 deficiency ({percent}), but the "
                f"model is not confident either way. Clinical judgement should lead."
            )
        return (
            f"The baseline phenotype pattern is not consistent with TANGO2 deficiency "
            f"({percent}). This does not exclude the diagnosis."
        )


@dataclass(slots=True)
class EvaluationReport:
    """Cross-validated performance, matching the metrics of Table 4."""

    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    brier: float
    brier_skill: float
    confusion: np.ndarray
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    calibration_predicted: np.ndarray
    calibration_observed: np.ndarray
    predicted_probabilities: np.ndarray
    true_labels: np.ndarray
    n_patients: int
    n_features: int
    provenance: Provenance

    def as_table(self) -> pd.DataFrame:
        """Metrics beside the paper's published values for comparison."""
        published = {
            "AUC": 0.994,
            "Accuracy": 0.982,
            "Sensitivity": 0.953,
            "Specificity": 1.000,
            "Brier score": 0.017,
        }
        observed = {
            "AUC": self.auc,
            "Accuracy": self.accuracy,
            "Sensitivity": self.sensitivity,
            "Specificity": self.specificity,
            "Brier score": self.brier,
        }
        return pd.DataFrame(
            {
                "Metric": list(published),
                "This model": [f"{observed[k]:.3f}" for k in published],
                "Published (real cohorts)": [f"{published[k]:.3f}" for k in published],
            }
        )


@dataclass(slots=True)
class DiagnosticModel:
    """A fitted Random Forest with its evaluation and provenance."""

    estimator: RandomForestClassifier
    feature_names: list[str]
    provenance: Provenance
    evaluation: EvaluationReport
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    model_version: str = "0.1.0"

    @property
    def feature_importances(self) -> pd.DataFrame:
        """Gini importances, descending."""
        frame = pd.DataFrame(
            {
                "Phenotype": self.feature_names,
                "Importance": self.estimator.feature_importances_,
            }
        )
        return frame.sort_values("Importance", ascending=False).reset_index(drop=True)

    def predict(
        self, present_phenotypes: list[str], pathognomonic: list[str] | None = None
    ) -> PatientPrediction:
        """Score one patient from the list of phenotypes they present.

        Parameters
        ----------
        present_phenotypes
            Labels of the baseline phenotypes observed. Unknown labels are
            ignored rather than raising, so that a caller working from a partial
            vocabulary still gets a prediction.
        pathognomonic
            Any pathognomonic markers observed. These are **not** model inputs
            and do not change the probability; they are carried through so the
            interface can surface them as independent referral triggers.
        """
        # A DataFrame rather than a bare array, so the column names match those
        # the forest was fitted with and scikit-learn does not warn.
        observed = set(present_phenotypes)
        vector = pd.DataFrame(
            [[1 if name in observed else 0 for name in self.feature_names]],
            columns=self.feature_names,
        )
        probability = float(self.estimator.predict_proba(vector)[0, 1])

        importances = dict(
            zip(self.feature_names, self.estimator.feature_importances_, strict=True)
        )
        present_importances = {
            name: importances[name] for name in present_phenotypes if name in importances
        }
        total = sum(present_importances.values())
        if total > 0:
            contributors = [
                Contribution(phenotype=name, weight=100.0 * value / total)
                for name, value in sorted(
                    present_importances.items(), key=lambda pair: pair[1], reverse=True
                )
            ][:6]
        else:
            contributors = []

        if probability > 0.8 or probability < 0.2:
            band = "High confidence"
        elif probability > 0.65 or probability < 0.35:
            band = "Moderate confidence"
        else:
            band = "Low confidence"

        return PatientPrediction(
            probability=probability,
            predicted_label="TANGO2 deficiency likely" if probability >= 0.5 else "TANGO2 unlikely",
            confidence_band=band,
            top_contributors=contributors,
            pathognomonic_present=list(pathognomonic or []),
            term_count=len(present_phenotypes),
            provenance=self.provenance,
        )


def _evaluate(
    estimator: RandomForestClassifier,
    features: pd.DataFrame,
    labels: np.ndarray,
    provenance: Provenance,
    *,
    n_repeats: int,
    seed: int,
) -> EvaluationReport:
    """Repeated stratified 10-fold cross-validation, as in Section 3.2.

    Every patient is predicted by a model that never saw them in training. A
    single split would be unreliable at this sample size, since a favourable
    split can produce a perfect score by chance.
    """
    # Repeats are run explicitly rather than via cross_val_predict, which only
    # accepts a partition. Each repeat is an independent 10-fold partition, so
    # every patient is predicted once per repeat by a model that never saw them;
    # averaging over repeats gives one stable out-of-fold estimate per patient.
    accumulated = np.zeros(len(labels))
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed + repeat)
        for train_idx, test_idx in splitter.split(features, labels):
            fold_model = clone(estimator)
            fold_model.fit(features.iloc[train_idx], labels[train_idx])
            accumulated[test_idx] += fold_model.predict_proba(features.iloc[test_idx])[:, 1]
    probabilities = accumulated / n_repeats

    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = matrix.ravel()

    base_rate = labels.mean()
    brier = brier_score_loss(labels, probabilities)
    reference_brier = float(np.mean((base_rate - labels) ** 2))

    fpr, tpr, _ = roc_curve(labels, probabilities)
    n_bins = min(10, max(3, len(np.unique(probabilities)) // 4))
    observed, predicted = calibration_curve(
        labels, probabilities, n_bins=n_bins, strategy="quantile"
    )

    return EvaluationReport(
        auc=float(roc_auc_score(labels, probabilities)),
        accuracy=float(accuracy_score(labels, predictions)),
        sensitivity=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        specificity=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        brier=float(brier),
        brier_skill=float(1.0 - brier / reference_brier) if reference_brier else float("nan"),
        confusion=matrix,
        roc_fpr=fpr,
        roc_tpr=tpr,
        calibration_predicted=predicted,
        calibration_observed=observed,
        predicted_probabilities=probabilities,
        true_labels=labels,
        n_patients=len(labels),
        n_features=features.shape[1],
        provenance=provenance,
    )


def train_model(
    cohort: SyntheticCohort | None = None,
    *,
    provenance: Provenance | None = None,
    n_estimators: int = 500,
    n_repeats: int = 3,
    seed: int = 0,
) -> DiagnosticModel:
    """Fit and cross-validate the diagnostic classifier.

    Parameters
    ----------
    cohort
        Training data. Defaults to a freshly generated synthetic cohort, in
        which case ``provenance`` is forced to ``"synthetic"``.
    n_repeats
        Cross-validation repeats. The paper uses 10; the default here is 3 so
        that an interactive application starts quickly.
    """
    if cohort is None:
        cohort = generate_synthetic_cohort(seed=seed)
        provenance = "synthetic"
    if provenance is None:
        provenance = "synthetic" if getattr(cohort, "is_synthetic", True) else "institutional"

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    evaluation = _evaluate(
        estimator,
        cohort.features,
        cohort.labels,
        provenance,
        n_repeats=n_repeats,
        seed=seed,
    )
    estimator.fit(cohort.features, cohort.labels)

    return DiagnosticModel(
        estimator=estimator,
        feature_names=[p.label for p in PHENOTYPES],
        provenance=provenance,
        evaluation=evaluation,
    )
