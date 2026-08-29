"""Phenotype vocabulary and the synthetic cohort generator.

The cohorts behind the paper were provided under institutional agreement by
Baylor College of Medicine and are **not redistributable**. This package
therefore ships two things instead of data: a fitting and evaluation harness
that reproduces the published analysis when pointed at the real cohorts, and the
synthetic generator below, which lets the software be demonstrated, tested and
deployed without any patient record.

Anything produced from synthetic data is labelled as such everywhere it appears.
A model fitted here describes the simulation and says nothing about real-world
accuracy.

Design of the generator
-----------------------
Sampling each phenotype as an independent coin flip at its published prevalence
would be wrong in a way that matters: it would make the classes almost perfectly
separable and produce a flattering, meaningless AUC. The paper's own finding is
that the signal is *distributed across many correlated phenotypes* rather than
carried by a few, so the generator uses a latent-burden model. Each simulated
patient draws a latent severity, and phenotype probabilities are shifted by it on
the logit scale. Phenotypes therefore co-occur within a patient, mildly affected
simulated patients present incompletely, and the resulting classification problem
has the overlapping structure the real one has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "PATHOGNOMONIC_MARKERS",
    "PHENOTYPES",
    "PHENOTYPE_CATEGORIES",
    "Phenotype",
    "SyntheticCohort",
    "generate_synthetic_cohort",
]


@dataclass(frozen=True, slots=True)
class Phenotype:
    """A baseline HPO term with its prevalence in each cohort."""

    label: str
    hpo_id: str
    tdd_prevalence: float
    """Percentage of TDD patients presenting this phenotype."""

    control_prevalence: float
    """Percentage of UDN controls presenting this phenotype."""

    published: bool
    """Whether the prevalences come from the paper or are plausible fill.

    Only the phenotypes of Table 5 and Figure 4 have published prevalences. The
    paper's shared-vocabulary set contains 31 terms but tabulates twelve, so the
    remainder carry estimated values used **only** to give the synthetic cohort
    realistic dimensionality. They are flagged here so that no figure derived
    from them can be mistaken for a published result.
    """

    category: str = "General"
    """Clinical grouping used to organise the intake checklist."""

    importance: float | None = None
    """Gini importance from Table 5, where the paper reports one."""


# Table 5 and Figure 4: the phenotypes with published prevalences.
_PUBLISHED: tuple[Phenotype, ...] = (
    Phenotype(
        "Poor speech",
        "HP:0002465",
        87.8,
        0.7,
        True,
        category="Speech and language",
        importance=0.137,
    ),
    Phenotype(
        "Motor delay", "HP:0001270", 90.0, 3.5, True, category="Motor and gait", importance=0.134
    ),
    Phenotype(
        "Dysarthria",
        "HP:0001260",
        90.0,
        3.5,
        True,
        category="Speech and language",
        importance=0.113,
    ),
    Phenotype(
        "Gait disturbance",
        "HP:0001288",
        87.8,
        2.8,
        True,
        category="Motor and gait",
        importance=0.107,
    ),
    Phenotype(
        "Slurred speech",
        "HP:0001350",
        82.2,
        0.7,
        True,
        category="Speech and language",
        importance=0.087,
    ),
    Phenotype(
        "EMG: myopathic abnormalities",
        "HP:0003458",
        82.2,
        1.4,
        True,
        category="Neuromuscular studies",
        importance=0.083,
    ),
    Phenotype(
        "Intellectual disability",
        "HP:0001249",
        87.8,
        7.1,
        True,
        category="Cognitive and developmental",
        importance=0.074,
    ),
    Phenotype(
        "Muscle weakness",
        "HP:0001324",
        84.4,
        3.5,
        True,
        category="Motor and gait",
        importance=0.074,
    ),
    Phenotype(
        "Delayed speech and language development",
        "HP:0000750",
        84.4,
        14.9,
        True,
        category="Speech and language",
        importance=0.043,
    ),
    Phenotype(
        "Gait ataxia", "HP:0002066", 82.2, 6.4, True, category="Motor and gait", importance=0.037
    ),
    Phenotype(
        "Global developmental delay",
        "HP:0001263",
        94.4,
        34.0,
        True,
        category="Cognitive and developmental",
        importance=0.029,
    ),
    Phenotype(
        "Abnormal EKG",
        "HP:0003115",
        40.0,
        1.4,
        True,
        category="Cardiac and systemic",
        importance=0.021,
    ),
)

# Additional shared-vocabulary terms, prevalences estimated. Synthetic use only.
_ESTIMATED: tuple[Phenotype, ...] = (
    Phenotype("Hypotonia", "HP:0001252", 71.1, 22.0, False, category="Motor and gait"),
    Phenotype("Seizure", "HP:0001250", 44.4, 28.4, False, category="Neurological"),
    Phenotype("Hypothyroidism", "HP:0000821", 37.8, 3.5, False, category="Cardiac and systemic"),
    Phenotype("Dysphagia", "HP:0002015", 46.7, 9.9, False, category="Speech and language"),
    Phenotype("Strabismus", "HP:0000486", 33.3, 12.1, False, category="Neurological"),
    Phenotype("Scoliosis", "HP:0002650", 28.9, 14.2, False, category="Motor and gait"),
    Phenotype(
        "Failure to thrive", "HP:0001508", 34.4, 19.9, False, category="Cognitive and developmental"
    ),
    Phenotype(
        "Microcephaly", "HP:0000252", 15.6, 21.3, False, category="Cognitive and developmental"
    ),
    Phenotype("Ataxia", "HP:0001251", 55.6, 9.2, False, category="Motor and gait"),
    Phenotype("Tremor", "HP:0001337", 24.4, 8.5, False, category="Neurological"),
    Phenotype("Nystagmus", "HP:0000639", 17.8, 11.3, False, category="Neurological"),
    Phenotype("Hyporeflexia", "HP:0001265", 31.1, 7.8, False, category="Neurological"),
    Phenotype("Muscle cramps", "HP:0003394", 26.7, 4.3, False, category="Motor and gait"),
    Phenotype("Exercise intolerance", "HP:0003546", 41.1, 5.7, False, category="Motor and gait"),
    Phenotype("Constipation", "HP:0002019", 32.2, 17.7, False, category="Cardiac and systemic"),
    Phenotype(
        "Feeding difficulties", "HP:0011968", 48.9, 24.8, False, category="Cardiac and systemic"
    ),
    Phenotype("Abnormal brain MRI", "HP:0012443", 38.9, 31.2, False, category="Neurological"),
    Phenotype("Short stature", "HP:0004322", 22.2, 18.4, False, category="Cardiac and systemic"),
    Phenotype(
        "Behavioural abnormality",
        "HP:0000708",
        36.7,
        22.7,
        False,
        category="Cognitive and developmental",
    ),
)

PHENOTYPES: tuple[Phenotype, ...] = _PUBLISHED + _ESTIMATED
"""The 31 shared-vocabulary baseline features used by the classifier."""

PHENOTYPE_CATEGORIES: tuple[str, ...] = (
    "Speech and language",
    "Motor and gait",
    "Cognitive and developmental",
    "Neurological",
    "Neuromuscular studies",
    "Cardiac and systemic",
)
"""Display order for the categories used to group the intake checklist."""

PATHOGNOMONIC_MARKERS: tuple[Phenotype, ...] = (
    Phenotype("Paroxysmal lethargy", "HP:0002329", 82.2, 0.0, True, category="Pathognomonic"),
    Phenotype("Compensatory head posture", "HP:0000643", 70.0, 0.0, True, category="Pathognomonic"),
)
"""Markers absent from every control in the source cohort.

Because they occur in zero UDN patients they are, by definition, not part of the
shared-vocabulary feature set, and they are **not** inputs to the classifier.
The paper reports them descriptively: both are bedside observations requiring no
test, and their absence across 141 phenotypically adjacent controls is what
motivates using them as immediate referral triggers. Observing zero false
positives in 141 controls bounds the rate only loosely -- by the rule of three,
the one-sided 95% upper bound is about 2.1%.
"""


@dataclass(frozen=True, slots=True)
class SyntheticCohort:
    """A simulated cohort. Never to be presented as patient data."""

    features: pd.DataFrame
    labels: np.ndarray
    is_synthetic: bool = True

    @property
    def n_tdd(self) -> int:
        """Number of simulated TDD patients."""
        return int(self.labels.sum())

    @property
    def n_control(self) -> int:
        """Number of simulated controls."""
        return int((1 - self.labels).sum())

    @property
    def term_counts(self) -> np.ndarray:
        """Baseline term burden per patient, the input to the severity scale."""
        return self.features.to_numpy().sum(axis=1)


def _calibrated_intercepts(targets: np.ndarray, latent_scale: float) -> np.ndarray:
    """Solve for intercepts whose marginal prevalence equals ``targets``.

    The latent burden shifts every phenotype, so the raw logit of a target
    prevalence does not reproduce that prevalence once the latent is averaged out.

    Because ``E[expit(b + Z)] != expit(b)`` for a random ``Z``, using the raw
    logit of the target prevalence would systematically distort the marginals --
    high prevalences get pulled down and low ones pushed up. Each intercept is
    therefore solved so that the *realised* marginal matches the published
    prevalence, using Gauss--Hermite quadrature over the latent and bisection
    over the intercept. Without this the generated cohort would not reproduce
    Table 5 even in expectation.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(64)
    weights = weights / weights.sum()
    latent = nodes * latent_scale

    def marginal(intercept: np.ndarray) -> np.ndarray:
        return _expit(intercept[:, None] + latent[None, :]) @ weights

    low = np.full(targets.shape, -20.0)
    high = np.full(targets.shape, 20.0)
    for _ in range(80):
        mid = 0.5 * (low + high)
        too_high = marginal(mid) > targets
        high = np.where(too_high, mid, high)
        low = np.where(too_high, low, mid)
    return 0.5 * (low + high)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_cohort(
    n_tdd: int = 90,
    n_control: int = 141,
    *,
    latent_scale: float = 1.1,
    seed: int = 0,
) -> SyntheticCohort:
    """Simulate a cohort matching the published prevalences and cohort sizes.

    Parameters
    ----------
    n_tdd, n_control
        Cohort sizes. Defaults match the paper's 90 and 141.
    latent_scale
        Standard deviation of the per-patient latent burden on the logit scale.
        Zero gives independent phenotypes and an unrealistically separable
        problem; larger values give stronger co-occurrence and more incompletely
        presenting patients. The default was chosen so that the marginal
        prevalences are preserved while phenotypes correlate within a patient.
    seed
        Seed for reproducibility.

    Returns
    -------
    SyntheticCohort
        Binary feature matrix, labels, and the synthetic flag.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.ones(n_tdd, dtype=int), np.zeros(n_control, dtype=int)])
    n_total = n_tdd + n_control

    tdd_rates = np.array([p.tdd_prevalence for p in PHENOTYPES]) / 100.0
    control_rates = np.array([p.control_prevalence for p in PHENOTYPES]) / 100.0

    tdd_intercepts = _calibrated_intercepts(tdd_rates, latent_scale)
    control_intercepts = _calibrated_intercepts(control_rates, latent_scale)
    base = np.where(labels[:, None] == 1, tdd_intercepts[None, :], control_intercepts[None, :])

    # A single latent burden per patient shifts every phenotype together, which
    # is what produces co-occurrence within a patient.
    latent = rng.normal(0.0, latent_scale, size=(n_total, 1))
    probabilities = _expit(base + latent)

    draws = rng.random((n_total, len(PHENOTYPES))) < probabilities
    frame = pd.DataFrame(draws.astype(int), columns=[p.label for p in PHENOTYPES])

    return SyntheticCohort(features=frame, labels=labels)
