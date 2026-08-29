"""Tests for the diagnostic classifier, synthetic cohort, and report generator."""

from __future__ import annotations

import numpy as np
import pytest

from tango2_forecast.ml import (
    PATHOGNOMONIC_MARKERS,
    PHENOTYPES,
    generate_synthetic_cohort,
    train_model,
)
from tango2_forecast.reports import ReportData, build_pdf


@pytest.fixture(scope="module")
def cohort():
    return generate_synthetic_cohort(seed=0)


@pytest.fixture(scope="module")
def model(cohort):
    return train_model(cohort, n_repeats=2)


class TestSyntheticCohort:
    def test_matches_published_cohort_sizes(self, cohort) -> None:
        assert cohort.n_tdd == 90
        assert cohort.n_control == 141

    def test_reproduces_published_prevalences(self, cohort) -> None:
        """The generator solves for intercepts so marginals match Table 5."""
        rates = cohort.features[cohort.labels == 1].mean() * 100
        published = [p for p in PHENOTYPES if p.published]
        for phenotype in published:
            assert rates[phenotype.label] == pytest.approx(phenotype.tdd_prevalence, abs=12.0), (
                phenotype.label
            )

    def test_term_burden_matches_the_paper(self, cohort) -> None:
        """TDD patients carry a median of 17 baseline terms in the source study."""
        tdd_counts = cohort.term_counts[cohort.labels == 1]
        assert np.median(tdd_counts) == pytest.approx(17, abs=3)

    def test_phenotypes_co_occur(self, cohort) -> None:
        """Independent sampling would make the problem unrealistically separable."""
        within_tdd = cohort.features[cohort.labels == 1].corr().to_numpy()
        off_diagonal = within_tdd[~np.eye(len(PHENOTYPES), dtype=bool)]
        assert np.nanmean(off_diagonal) > 0.02

    def test_is_reproducible(self) -> None:
        a = generate_synthetic_cohort(seed=7)
        b = generate_synthetic_cohort(seed=7)
        assert a.features.equals(b.features)

    def test_pathognomonic_markers_are_absent_from_controls(self) -> None:
        """Zero control prevalence is why they cannot be shared-vocabulary features."""
        for marker in PATHOGNOMONIC_MARKERS:
            assert marker.control_prevalence == 0.0

    def test_pathognomonic_markers_are_not_model_features(self) -> None:
        labels = {p.label for p in PHENOTYPES}
        for marker in PATHOGNOMONIC_MARKERS:
            assert marker.label not in labels


class TestClassifier:
    def test_feature_count_matches_the_shared_vocabulary(self, model) -> None:
        assert len(model.feature_names) == 31

    def test_synthetic_provenance_is_recorded(self, model) -> None:
        """A synthetic model must never be presentable as an institutional one."""
        assert model.provenance == "synthetic"
        assert model.evaluation.provenance == "synthetic"

    def test_discriminates_on_the_simulated_problem(self, model) -> None:
        assert model.evaluation.auc > 0.9

    def test_is_calibrated(self, model) -> None:
        assert model.evaluation.brier < 0.1
        assert model.evaluation.brier_skill > 0.5

    def test_typical_presentation_scores_high(self, model) -> None:
        prediction = model.predict(
            [
                "Dysarthria",
                "Motor delay",
                "Gait disturbance",
                "Poor speech",
                "Intellectual disability",
                "Muscle weakness",
                "Slurred speech",
            ]
        )
        assert prediction.probability > 0.5
        assert "TANGO2" in prediction.plain_language()

    def test_unrelated_presentation_scores_low(self, model) -> None:
        assert model.predict(["Microcephaly", "Short stature"]).probability < 0.5

    def test_unknown_phenotypes_are_ignored_rather_than_raising(self, model) -> None:
        model.predict(["Not a real phenotype"])

    def test_pathognomonic_markers_do_not_change_the_probability(self, model) -> None:
        """They are reported separately, never fed to the model."""
        observed = ["Dysarthria", "Motor delay"]
        without = model.predict(observed)
        with_marker = model.predict(observed, pathognomonic=["Paroxysmal lethargy"])
        assert without.probability == with_marker.probability
        assert with_marker.pathognomonic_present == ["Paroxysmal lethargy"]

    def test_interpretation_never_claims_to_exclude_a_diagnosis(self, model) -> None:
        text = model.predict(["Microcephaly"]).plain_language()
        assert "does not exclude" in text.lower()


class TestReportGeneration:
    def _data(self, **overrides) -> ReportData:
        base = {
            "patient_reference": "TEST-1",
            "clinician": "Dr. Test",
            "phenotypes": ["Dysarthria", "Motor delay"],
            "pathognomonic_markers": [],
            "probability": 0.91,
            "predicted_label": "TANGO2 deficiency likely",
            "confidence_band": "High confidence",
            "interpretation": "Strongly consistent with TANGO2 deficiency.",
            "top_contributors": [("Dysarthria", 0.113)],
            "model_provenance": "synthetic",
            "model_version": "0.1.0",
            "auc": 0.99,
            "sensitivity": 0.95,
            "specificity": 1.0,
            "brier": 0.02,
        }
        return ReportData(**{**base, **overrides})

    def test_produces_a_valid_pdf(self) -> None:
        pdf = build_pdf(self._data())
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 3000

    def test_forecast_section_is_optional(self) -> None:
        without = build_pdf(self._data())
        with_forecast = build_pdf(
            self._data(
                include_forecast=True,
                beta=0.016,
                beta_critical=0.0127,
                regime="recurrent",
                regime_note="Above the critical amplitude.",
                mean_crisis_count=4.9,
                first_crisis_day=232.0,
                probability_any_crisis=1.0,
                n_realizations=50,
                seed=0,
            )
        )
        assert len(with_forecast) > len(without)

    def test_synthetic_warning_appears_in_the_document(self) -> None:
        pypdf = pytest.importorskip("pypdf")
        from io import BytesIO

        reader = pypdf.PdfReader(BytesIO(build_pdf(self._data())))
        text = " ".join(page.extract_text() for page in reader.pages)
        assert "SYNTHETIC DATA" in text
        assert "not a medical device" in text.lower()

    def test_scope_statement_survives_an_empty_assessment(self) -> None:
        pdf = build_pdf(self._data(phenotypes=[], top_contributors=[]))
        assert pdf.startswith(b"%PDF")
