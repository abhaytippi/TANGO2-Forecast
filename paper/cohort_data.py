"""Loading and feature construction for the restricted Baylor cohorts.

The two spreadsheets are NOT part of this repository. Place them in ``data/``
(or point the TANGO2_DATA environment variable at their folder):

    data/Copy of TANGO2_Dataset.xlsx   (90 TDD patients)
    data/UDN_diagnoses.xlsx            (141 UDN controls)

Phenotypes are read from the "HPO Term Labels" column, split on commas.
"""

from __future__ import annotations

import collections
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("TANGO2_DATA", os.path.join(ROOT, "data"))
TDD_FILE = os.path.join(DATA, "Copy of TANGO2_Dataset.xlsx")
UDN_FILE = os.path.join(DATA, "UDN_diagnoses.xlsx")

# Crisis defining phenotypes (paper Section 3.2), matched on the HPO label.
CRISIS_KEYWORDS = [
    "rhabdo", "myoglobin", "creatine kinase", "cardiac arrest", "cardiomyopathy", "arrhythmi", "qt",
    "metabolic acidosis", "lactic acid", "lactate", "hyperammon", "hypoglyc", "acute encephalopath",
    "coma", "tachycard", "ketoacidosis", "ketosis",
]


def is_crisis(term: str) -> bool:
    t = term.lower()
    return any(k in t for k in CRISIS_KEYWORDS) and "glaucoma" not in t


def is_ecg(term: str) -> bool:
    t = term.lower()
    return "ekg" in t or "ecg" in t or "electrocardiogra" in t


def _terms(df: pd.DataFrame) -> list[set[str]]:
    return [set(s.strip() for s in str(v).split(",") if s.strip()) for v in df["HPO Term Labels"]]


def load():
    for f in (TDD_FILE, UDN_FILE):
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    return _terms(pd.read_excel(TDD_FILE)), _terms(pd.read_excel(UDN_FILE))


def shared_features(tdd, udn, min_count: int = 3) -> list[str]:
    """Terms in >= min_count patients, used by both cohorts, not crisis, not ECG (30 features)."""
    c = collections.Counter(x for p in tdd + udn for x in p)
    ct = collections.Counter(x for p in tdd for x in p)
    cu = collections.Counter(x for p in udn for x in p)
    return sorted(x for x in c if c[x] >= min_count and ct[x] and cu[x] and not is_crisis(x) and not is_ecg(x))


def design(features, tdd, udn):
    X = np.array([[1 if f in p else 0 for f in features] for p in tdd + udn])
    y = np.array([1] * len(tdd) + [0] * len(udn))
    return X, y


def load_tdd_term_counts() -> np.ndarray:
    """Number of retained (non crisis) baseline terms per TDD patient, used for severity."""
    tdd, _ = load()
    return np.array([sum(1 for t in p if not is_crisis(t)) for p in tdd])
