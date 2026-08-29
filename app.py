"""TANGO2-Forecast web application.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tango2_forecast import __version__  # noqa: E402
from tango2_forecast.forecast import (  # noqa: E402
    BETA_MAX,
    BETA_MIN,
    CrisisModelParameters,
    GridSpec,
    IMEXCrisisSolver,
    SeverityScale,
    bifurcation_curve,
    classify_regime,
    critical_amplitude,
)
from tango2_forecast.ml import (  # noqa: E402
    PATHOGNOMONIC_MARKERS,
    PHENOTYPE_CATEGORIES,
    PHENOTYPES,
    train_model,
)
from tango2_forecast.reports import ReportData, build_pdf  # noqa: E402
from tango2_forecast.ui import figures  # noqa: E402
from tango2_forecast.ui.theme import CUSTOM_CSS, register_plotly_template  # noqa: E402

st.set_page_config(
    page_title="TANGO2-Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
register_plotly_template()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached work
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the diagnostic model")
def load_model():
    """Fit the classifier once per session."""
    return train_model(n_repeats=3)


@st.cache_data(show_spinner="Locating the model threshold")
def load_bifurcation():
    """Compute the critical amplitude and the steady-state branch."""
    beta_star = critical_amplitude()
    betas, peaks = bifurcation_curve(beta_values=np.linspace(0.008, 0.020, 61))
    return beta_star, betas, peaks


@st.cache_data(show_spinner="Running the forecast")
def run_forecast(
    beta: float, horizon_days: float, n_realizations: int, seed: int, store_field: bool
):
    """Run the crisis model. Cached on its arguments."""
    grid = GridSpec(horizon_days=horizon_days)
    solver = IMEXCrisisSolver(CrisisModelParameters(), grid)
    return solver.run(
        beta=beta,
        n_realizations=n_realizations,
        seed=seed,
        store_field=store_field,
        field_stride=max(1, int(grid.n_steps / 200)),
    )


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------


def app_header() -> None:
    """Render the fixed identification bar at the top of every page."""
    model = load_model()
    status = "Development dataset" if model.provenance == "synthetic" else "Validated dataset"
    st.markdown(
        '<div class="app-header">'
        '<span class="name">TANGO2-Forecast</span>'
        f'<span class="tag">{status}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str = "") -> None:
    """Page heading."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def notice(text: str, kind: str = "neutral") -> None:
    """Render a quiet, bordered note. Used sparingly."""
    st.markdown(f'<div class="notice notice-{kind}">{text}</div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, note: str = "", kind: str = "") -> None:
    """Render a single figure in a bordered panel."""
    css_class = f"stat is-{kind}" if kind else "stat"
    st.markdown(
        f'<div class="{css_class}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def go_to(page: str) -> None:
    """Request navigation to another page on the next run."""
    st.session_state["pending_nav"] = page
    st.rerun()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def page_overview() -> None:
    """Landing page."""
    page_title("Overview")

    st.markdown(
        "TANGO2 deficiency disorder is a rare autosomal recessive condition. "
        "Between crises, patients present with a stable neurodevelopmental "
        "phenotype: developmental delay, intellectual disability, dysarthria, "
        "and gait disturbance. During a crisis, typically triggered by fasting "
        "or febrile illness, patients decompensate acutely with rhabdomyolysis, "
        "metabolic acidosis, and ventricular arrhythmia."
    )
    st.markdown(
        "Because the interictal phenotype overlaps with many neurogenetic "
        "conditions, the disorder is often not suspected until a crisis has "
        "already occurred. Prophylactic treatment, B complex supplementation "
        "with fasting avoidance, reduces crisis frequency substantially when "
        "started before that point."
    )

    st.markdown("## What this tool provides")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Diagnostic assessment")
        st.markdown(
            "A baseline phenotype checklist produces a probability that a "
            "patient's presentation is consistent with TANGO2 deficiency, drawn "
            "from a classifier validated by repeated cross validation on a "
            "reference cohort."
        )
    with right:
        st.markdown("#### Crisis forecasting")
        st.markdown(
            "A mechanistic model estimates how a patient's baseline severity "
            "relates to the expected timing and frequency of metabolic crises, "
            "and where that patient sits relative to a modelled stability "
            "threshold."
        )

    st.markdown("## Referral pathway")
    st.markdown(
        "Two findings support an immediate referral for TANGO2 sequencing "
        "regardless of the assessment score: paroxysmal lethargy and "
        "compensatory head posture. Both are bedside observations that require "
        "no laboratory testing. A broader neurological pattern, dysarthria "
        "combined with gait disturbance and intellectual disability, supports "
        "genetic evaluation with TANGO2 included in the differential."
    )

    st.markdown("---")
    st.markdown("Begin with **Patient Assessment** in the sidebar.")


def page_assessment() -> None:
    """Intake and diagnostic assessment."""
    page_title("Patient Assessment", "Baseline phenotype screening")
    model = load_model()

    st.markdown("### Patient information")
    id_col, age_col, sex_col = st.columns(3)
    reference = id_col.text_input("Patient reference", placeholder="MRN or study ID")
    age = age_col.text_input("Age", placeholder="e.g. 4 years")
    sex = sex_col.selectbox("Sex", ["", "Female", "Male", "Other / not specified"])

    height_col, weight_col, _ = st.columns(3)
    height = height_col.text_input("Height", placeholder="cm")
    weight = weight_col.text_input("Weight", placeholder="kg")

    uploads = st.file_uploader(
        "Supporting documents (labs, ECG, prior genetic testing, clinical notes)",
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "csv", "txt"],
    )

    st.markdown("### Referral markers")
    st.caption(
        "Present in a majority of confirmed cases and in no reference control. "
        "These are not inputs to the model below and stand on their own."
    )
    marker_cols = st.columns(2)
    observed_markers = [
        marker.label
        for marker, column in zip(PATHOGNOMONIC_MARKERS, marker_cols, strict=True)
        if column.checkbox(marker.label, key=f"marker_{marker.hpo_id}")
    ]
    if observed_markers:
        notice(
            f"<strong>Referral marker present.</strong> Observed: "
            f"{', '.join(observed_markers)}. This finding supports proceeding "
            "directly to genetic testing independent of the score below.",
            "alert",
        )

    st.markdown("### Clinical findings")
    st.caption(
        "Select every baseline finding present. Findings occurring only during "
        "an acute crisis are intentionally excluded from this checklist."
    )

    selected: list[str] = []
    by_category: dict[str, list] = {c: [] for c in PHENOTYPE_CATEGORIES}
    for phenotype in PHENOTYPES:
        by_category.setdefault(phenotype.category, []).append(phenotype)

    for category in PHENOTYPE_CATEGORIES:
        items = by_category.get(category, [])
        if not items:
            continue
        with st.expander(category, expanded=(category == "Speech and language")):
            cols = st.columns(2)
            for index, phenotype in enumerate(items):
                if cols[index % 2].checkbox(phenotype.label, key=f"ph_{phenotype.hpo_id}"):
                    selected.append(phenotype.label)

    st.divider()

    if not selected:
        notice("Select at least one clinical finding to generate an assessment.", "neutral")
        return

    prediction = model.predict(selected, observed_markers)
    st.session_state["assessment"] = {
        "reference": reference,
        "age": age,
        "sex": sex,
        "height": height,
        "weight": weight,
        "documents": [f.name for f in (uploads or [])],
        "selected": selected,
        "markers": observed_markers,
        "probability": prediction.probability,
        "term_count": len(selected),
        "predicted_label": prediction.predicted_label,
        "confidence_band": prediction.confidence_band,
        "interpretation": prediction.plain_language(),
        "contributors": [(c.phenotype, c.weight) for c in prediction.top_contributors],
    }

    st.markdown("### Assessment")
    left, right = st.columns([1, 1.4])
    with left:
        kind = (
            "alert"
            if prediction.probability >= 0.8
            else "caution"
            if prediction.probability >= 0.5
            else ""
        )
        stat_card(
            "Assessment probability",
            f"{prediction.probability:.0%}",
            prediction.confidence_band,
            kind,
        )
        stat_card("Baseline findings recorded", f"{len(selected)} of {len(PHENOTYPES)}")
    with right:
        st.markdown(f"#### {prediction.predicted_label}")
        st.markdown(prediction.plain_language())
        if prediction.top_contributors:
            st.markdown("**Findings contributing most to this assessment**")
            table = pd.DataFrame(
                [
                    {"Finding": c.phenotype, "Relative contribution": f"{c.weight:.0f}%"}
                    for c in prediction.top_contributors
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption(
                "Relative contribution is each finding's share of the combined "
                "model weight of the findings recorded above, and sums to 100 "
                "percent across this list."
            )

    if prediction.probability >= 0.5 or observed_markers:
        st.markdown("")
        if st.button("Continue to Crisis Forecast", type="primary"):
            go_to("Crisis Forecast")


def page_forecast() -> None:
    """Crisis model with clinician-facing controls."""
    page_title("Crisis Forecast", "Estimated crisis timing and frequency")

    beta_star, betas, peaks = load_bifurcation()
    assessment = st.session_state.get("assessment")
    scale = SeverityScale.fit([1, len(PHENOTYPES)])

    st.markdown("### Severity input")
    options = ["From patient assessment", "Mild", "Moderate", "Severe", "Custom"]
    default_mode = "From patient assessment" if assessment else "Moderate"
    mode = st.radio(
        "Set baseline severity by",
        options,
        index=options.index(default_mode),
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "From patient assessment" and assessment:
        beta = float(scale.beta(assessment["term_count"]))
        st.caption(f"Using {assessment['term_count']} findings recorded in Patient Assessment.")
    elif mode == "Mild":
        beta = BETA_MIN + 0.15 * (BETA_MAX - BETA_MIN)
    elif mode == "Moderate":
        beta = BETA_MIN + 0.5 * (BETA_MAX - BETA_MIN)
    elif mode == "Severe":
        beta = BETA_MIN + 0.85 * (BETA_MAX - BETA_MIN)
    else:
        severity_index = st.slider(
            "Severity index",
            0,
            100,
            50,
            help="0 corresponds to the mildest calibrated presentation, 100 to the most severe.",
        )
        beta = BETA_MIN + (severity_index / 100.0) * (BETA_MAX - BETA_MIN)

    if mode == "From patient assessment" and not assessment:
        st.caption("Complete a Patient Assessment to set this automatically.")

    with st.expander("Simulation settings"):
        horizon_days = st.select_slider(
            "Forecast horizon",
            [180, 365, 730, 900, 1095],
            value=900,
            help="Days from baseline assessment.",
        )
        n_realizations = st.select_slider(
            "Simulated patients",
            [20, 50, 100, 200],
            value=50,
            help="More simulated patients give a smoother range of outcomes.",
        )
        seed = st.number_input("Random seed", 0, 9999, 0)
        st.caption(
            f"Underlying model parameter (beta): {beta:.5f}  |  "
            f"Model threshold (beta*): {beta_star:.5f}"
        )

    assessment_result = classify_regime(beta, beta_star)
    result = run_forecast(beta, float(horizon_days), int(n_realizations), int(seed), True)
    summary = result.summary()

    st.markdown("### Forecast summary")
    columns = st.columns(4)
    with columns[0]:
        kind = "alert" if assessment_result.regime == "recurrent" else ""
        stat_card(
            "Risk classification",
            assessment_result.regime.title(),
            f"{assessment_result.relative_margin:+.0%} relative to threshold",
            kind,
        )
    with columns[1]:
        stat_card(
            "Expected crises",
            f"{summary['mean_crisis_count']:.1f}",
            f"over {horizon_days} days, plus or minus {summary['sd_crisis_count']:.1f}",
        )
    with columns[2]:
        day = summary["mean_first_crisis_day"]
        stat_card(
            "Expected first event",
            "Beyond horizon" if np.isnan(day) else f"Day {day:.0f}",
            "" if np.isnan(day) else f"plus or minus {summary['sd_first_crisis_day']:.0f} days",
        )
    with columns[3]:
        stat_card(
            "Simulated patients affected",
            f"{summary['probability_any_crisis']:.0%}",
            f"of {result.n_realizations} simulated",
        )

    st.session_state["forecast"] = {
        "beta": beta,
        "beta_critical": beta_star,
        "regime": assessment_result.regime,
        "regime_note": assessment_result.clinical_note(),
        "mean_crisis_count": summary["mean_crisis_count"],
        "first_crisis_day": (
            None if np.isnan(summary["mean_first_crisis_day"]) else summary["mean_first_crisis_day"]
        ),
        "probability_any_crisis": summary["probability_any_crisis"],
        "n_realizations": result.n_realizations,
        "seed": int(seed),
    }

    notice(
        assessment_result.clinical_note(),
        "caution" if assessment_result.regime == "quiescent" else "alert",
    )

    tabs = st.tabs(["Projected range", "Single trajectory", "Risk distribution", "Model threshold"])
    with tabs[0]:
        st.plotly_chart(figures.ensemble_figure(result), width="stretch")
        st.caption(
            "Shaded bands show the range of outcomes across simulated patients "
            "with the same baseline severity. The range widens with time."
        )
    with tabs[1]:
        st.plotly_chart(figures.trajectory_figure(result), width="stretch")
    with tabs[2]:
        st.plotly_chart(figures.spacetime_figure(result), width="stretch")
        st.caption("Shows where risk concentrates within the modelled physiological system.")
    with tabs[3]:
        st.plotly_chart(
            figures.bifurcation_figure(betas, peaks, beta_star, result.parameters.r_star, beta),
            width="stretch",
        )


def page_insights() -> None:
    """Model performance."""
    page_title("Model Insights", "Diagnostic classifier performance")
    model = load_model()
    report = model.evaluation

    if model.provenance == "synthetic":
        st.caption(
            "Trained on a simulated development cohort built from published "
            "prevalence data. Figures below describe this development model and "
            "will be superseded once the classifier is trained on the "
            "institutional cohort."
        )

    columns = st.columns(4)
    for column, (label, value) in zip(
        columns,
        [
            ("Discrimination (AUC)", f"{report.auc:.3f}"),
            ("Sensitivity", f"{report.sensitivity:.0%}"),
            ("Specificity", f"{report.specificity:.0%}"),
            ("Calibration (Brier)", f"{report.brier:.3f}"),
        ],
        strict=True,
    ):
        with column:
            stat_card(label, value, "Cross validated")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(figures.roc_figure(report), width="stretch")
    with right:
        st.plotly_chart(figures.calibration_figure(report), width="stretch")

    importances = model.feature_importances
    st.plotly_chart(
        figures.importance_figure(
            importances["Phenotype"].tolist(), importances["Importance"].tolist()
        ),
        width="stretch",
    )
    st.caption(
        "Values shown here describe the model overall, and differ from the "
        "relative contribution shown on an individual assessment, which is "
        "specific to the findings recorded for that patient."
    )

    st.markdown("### Performance detail")
    st.dataframe(report.as_table(), width="stretch", hide_index=True)


def page_report() -> None:
    """Render the exportable clinical report."""
    page_title("Clinical Report")
    assessment = st.session_state.get("assessment")
    if not assessment:
        notice("Complete a Patient Assessment first.", "neutral")
        return

    model = load_model()
    forecast = st.session_state.get("forecast")

    left, right = st.columns(2)
    reference = left.text_input("Patient reference", value=assessment.get("reference", ""))
    clinician = right.text_input("Assessing clinician")

    include_forecast = st.checkbox(
        "Include crisis forecast section", value=forecast is not None, disabled=forecast is None
    )

    data = ReportData(
        patient_reference=reference,
        clinician=clinician,
        phenotypes=assessment["selected"],
        pathognomonic_markers=assessment["markers"],
        probability=assessment["probability"],
        predicted_label=assessment["predicted_label"],
        confidence_band=assessment["confidence_band"],
        interpretation=assessment["interpretation"],
        top_contributors=assessment["contributors"],
        model_provenance=model.provenance,
        model_version=model.model_version,
        auc=model.evaluation.auc,
        sensitivity=model.evaluation.sensitivity,
        specificity=model.evaluation.specificity,
        brier=model.evaluation.brier,
        patient_age=assessment.get("age", ""),
        patient_sex=assessment.get("sex", ""),
        patient_height=assessment.get("height", ""),
        patient_weight=assessment.get("weight", ""),
        attached_documents=assessment.get("documents", []),
        include_forecast=bool(include_forecast and forecast),
        **(
            {
                "beta": forecast["beta"],
                "beta_critical": forecast["beta_critical"],
                "regime": forecast["regime"],
                "regime_note": forecast["regime_note"],
                "mean_crisis_count": forecast["mean_crisis_count"],
                "first_crisis_day": forecast["first_crisis_day"],
                "probability_any_crisis": forecast["probability_any_crisis"],
                "n_realizations": forecast["n_realizations"],
                "seed": forecast["seed"],
            }
            if include_forecast and forecast
            else {}
        ),
    )

    st.divider()
    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Download PDF report",
            build_pdf(data),
            file_name=f"tango2_assessment_{reference or 'unreferenced'}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with download_right:
        rows = [
            ("Patient reference", reference or "Not recorded"),
            ("Assessing clinician", clinician or "Not recorded"),
            ("Findings recorded", "; ".join(assessment["selected"])),
            ("Referral markers", "; ".join(assessment["markers"]) or "None observed"),
            ("Assessment probability", f"{assessment['probability']:.3f}"),
            ("Classification", assessment["predicted_label"]),
        ]
        frame = pd.DataFrame(rows, columns=["Field", "Value"])
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode(),
            file_name=f"tango2_assessment_{reference or 'unreferenced'}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.dataframe(frame, width="stretch", hide_index=True)


def page_documentation() -> None:
    """In-app documentation."""
    page_title("Documentation")

    st.markdown("### Intended use")
    st.markdown(
        "This tool supports the decision to refer a patient for TANGO2 genetic "
        "sequencing, based on a baseline clinical presentation. It does not "
        "confirm or exclude a diagnosis on its own. Molecular testing remains "
        "the diagnostic standard."
    )

    st.markdown("### Model assumptions")
    st.markdown(
        "- Findings that occur only during an acute metabolic crisis are "
        "excluded from the checklist, so the assessment reflects a baseline "
        "presentation rather than crisis history.\n"
        "- The checklist is restricted to findings recorded consistently across "
        "the reference cohorts used to build the classifier.\n"
        "- Severity in the crisis forecast is derived from the number of "
        "findings recorded relative to a reference population, and does not "
        "account for current treatment.\n"
        "- The forecast does not represent specific triggers such as fasting or "
        "illness individually; their combined effect is represented as a "
        "general variability term."
    )

    st.markdown("### Reading the assessment probability")
    guide = pd.DataFrame(
        [
            (
                "Above 80%",
                "Strongly consistent with TANGO2 deficiency. Supports referral for sequencing.",
            ),
            (
                "50% to 80%",
                "Moderately consistent. Genetic evaluation with TANGO2 in the differential "
                "is reasonable.",
            ),
            (
                "20% to 50%",
                "Not typical, but not ruled out by the model. Clinical judgement leads.",
            ),
            ("Below 20%", "Not consistent with TANGO2 deficiency. Does not exclude the diagnosis."),
        ],
        columns=["Assessment probability", "Reading"],
    )
    st.dataframe(guide, width="stretch", hide_index=True)
    st.caption(
        "A referral marker should be acted on regardless of the probability "
        "shown. Those markers are not inputs to the assessment score."
    )

    st.markdown("### Reading the crisis forecast")
    st.markdown(
        "The forecast places a patient's baseline severity relative to a "
        "modelled stability threshold. Above the threshold, crises recur in the "
        "model. Below it, the model produces a stable state, though random "
        "variability can still produce an occasional crisis in that range. A "
        "classification below the threshold should be read as lower "
        "likelihood, not as absence of risk."
    )

    st.markdown("### References")
    st.markdown(
        "- Lalani, S. R., Liu, P., Rosenfeld, J. A., et al. (2016). Recurrent "
        "muscle weakness with rhabdomyolysis, metabolic crises, and cardiac "
        "arrhythmia due to bi-allelic TANGO2 mutations. American Journal of "
        "Human Genetics, 98(2), 347 to 357.\n"
        "- Dines, J. N., Golden-Grant, K., LaCroix, A., et al. (2019). TANGO2 "
        "deficiency: expanding the clinical phenotype and spectrum of disease. "
        "Genetics in Medicine, 21(3), 601 to 607.\n"
        "- Kohler, S., Carmody, L., Vasilevsky, N., et al. (2019). Expansion of "
        "the Human Phenotype Ontology knowledge base and resources. Nucleic "
        "Acids Research, 47(D1), D1018 to D1027."
    )


PAGES = {
    "Overview": page_overview,
    "Patient Assessment": page_assessment,
    "Crisis Forecast": page_forecast,
    "Model Insights": page_insights,
    "Clinical Report": page_report,
    "Documentation": page_documentation,
}


def main() -> None:
    """Render the selected page."""
    app_header()
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        st.session_state["nav_choice"] = target
        st.session_state["nav_radio"] = target

    with st.sidebar:
        st.markdown("#### TANGO2-Forecast")
        st.session_state.setdefault("nav_radio", "Overview")
        choice = st.radio(
            "Navigate",
            list(PAGES),
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.session_state["nav_choice"] = choice
        st.divider()

    PAGES[st.session_state["nav_choice"]]()

    with st.sidebar:
        st.divider()
        st.markdown(
            f'<div class="sidebar-foot">Version {__version__}<br>Created by Abhay Tippimath</div>',
            unsafe_allow_html=True,
        )


main()
