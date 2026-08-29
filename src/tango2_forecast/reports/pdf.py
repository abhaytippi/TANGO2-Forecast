"""Clinical report generation (Module C).

Produces a document a clinician could reasonably place in a research file: a
header block identifying the software and its version, the assessment, its
interpretation, and, inseparably, the boundary of what the assessment is worth.

The scope statements are not an appendix. They are laid out with the same visual
weight as the results, because a report that travels without them is the failure
mode this software most needs to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..disclaimer import (
    CLASSIFIER_SCOPE,
    FORECAST_RESEARCH_USE_ONLY,
    NOT_A_MEDICAL_DEVICE,
    PERFORMANCE_TRANSFER_CAVEAT,
    SYNTHETIC_DATA_WARNING,
    TREATMENT_NAIVE_NOTE,
)

__all__ = ["ReportData", "build_pdf"]

INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#E2E8F0")
PRIMARY = colors.HexColor("#1D4ED8")
DANGER = colors.HexColor("#B91C1C")
DANGER_SOFT = colors.HexColor("#FEE2E2")
AMBER_SOFT = colors.HexColor("#FEF3C7")
CANVAS = colors.HexColor("#F8FAFC")


@dataclass(slots=True)
class ReportData:
    """Everything that appears in a report.

    Assembled by the caller so that this module has no dependency on the
    dashboard, the classifier, or the solver, and can be unit tested alone.
    """

    patient_reference: str
    clinician: str
    phenotypes: list[str]
    pathognomonic_markers: list[str]
    probability: float
    predicted_label: str
    confidence_band: str
    interpretation: str
    top_contributors: list[tuple[str, float]]
    model_provenance: str
    model_version: str
    auc: float
    sensitivity: float
    specificity: float
    brier: float

    # Optional demographic context. Not used by the model; recorded for the
    # clinical record only.
    patient_age: str = ""
    patient_sex: str = ""
    patient_height: str = ""
    patient_weight: str = ""
    attached_documents: list[str] = field(default_factory=list)

    # Crisis model section. Omitted entirely when the forecast was not run.
    include_forecast: bool = False
    beta: float | None = None
    beta_critical: float | None = None
    regime: str | None = None
    regime_note: str | None = None
    mean_crisis_count: float | None = None
    first_crisis_day: float | None = None
    probability_any_crisis: float | None = None
    n_realizations: int | None = None
    seed: int | None = None

    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=18,
            textColor=INK,
            spaceAfter=2,
            alignment=0,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9.5, textColor=MUTED, spaceAfter=10
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=12,
            textColor=INK,
            spaceBefore=14,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontSize=9,
            textColor=MUTED,
            spaceBefore=8,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8, leading=11, textColor=MUTED
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["Normal"], fontSize=8.5, leading=12, textColor=INK
        ),
    }


def _callout(text: str, style: ParagraphStyle, background, border) -> Table:
    """Build a bordered, filled block used for scope statements."""
    table = Table([[Paragraph(text, style)]], colWidths=[6.9 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.75, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _field_table(rows: list[tuple[str, str]]) -> Table:
    """Two-column label/value table."""
    table = Table(rows, colWidths=[2.1 * inch, 4.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8.5),
                ("FONT", (1, 0), (1, -1), "Helvetica", 8.5),
                ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
                ("TEXTCOLOR", (1, 0), (1, -1), INK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _page_furniture(canvas, doc) -> None:
    """Header rule and footer on every page."""
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(0.85 * inch, 10.55 * inch, 7.75 * inch, 10.55 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.85 * inch, 10.65 * inch, "TANGO2-FORECAST  |  RESEARCH SOFTWARE")
    canvas.drawRightString(7.75 * inch, 10.65 * inch, "NOT A MEDICAL DEVICE")
    canvas.line(0.85 * inch, 0.72 * inch, 7.75 * inch, 0.72 * inch)
    canvas.drawString(
        0.85 * inch,
        0.55 * inch,
        "Supports a referral decision. Does not confirm or exclude a diagnosis.",
    )
    canvas.drawRightString(7.75 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(data: ReportData) -> bytes:
    """Render a report and return the PDF bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.75 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.9 * inch,
        title="TANGO2-Forecast Assessment Report",
        author="TANGO2-Forecast",
    )
    s = _styles()
    story: list = []

    # --- Header ---------------------------------------------------------
    story.append(Paragraph("TANGO2 Deficiency Assessment", s["title"]))
    story.append(
        Paragraph(
            f"Generated {data.generated_at} &nbsp;|&nbsp; TANGO2-Forecast v{data.model_version}",
            s["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))

    story.append(
        _callout(
            f"<b>Not a medical device.</b> {NOT_A_MEDICAL_DEVICE}",
            s["callout"],
            AMBER_SOFT,
            colors.HexColor("#FCD34D"),
        )
    )
    story.append(Spacer(1, 8))

    if data.model_provenance.lower() == "synthetic":
        story.append(
            _callout(
                f"<b>Synthetic model.</b> {SYNTHETIC_DATA_WARNING}",
                s["callout"],
                DANGER_SOFT,
                colors.HexColor("#FCA5A5"),
            )
        )
        story.append(Spacer(1, 8))

    # --- Patient summary ------------------------------------------------
    story.append(Paragraph("Patient summary", s["h2"]))
    demographic_rows = [
        ("Patient reference", data.patient_reference or "Not recorded"),
        ("Assessing clinician", data.clinician or "Not recorded"),
    ]
    if data.patient_age or data.patient_sex or data.patient_height or data.patient_weight:
        demographic_rows.append(
            (
                "Age / sex",
                f"{data.patient_age or 'Not recorded'} / {data.patient_sex or 'Not recorded'}",
            )
        )
        if data.patient_height or data.patient_weight:
            demographic_rows.append(
                (
                    "Height / weight",
                    f"{data.patient_height or 'Not recorded'}"
                    f" / {data.patient_weight or 'Not recorded'}",
                )
            )
    demographic_rows.append(("Baseline terms recorded", str(len(data.phenotypes))))
    demographic_rows.append(
        (
            "Baseline phenotypes",
            ", ".join(data.phenotypes) if data.phenotypes else "None recorded",
        )
    )
    if data.attached_documents:
        demographic_rows.append(("Supporting documents", ", ".join(data.attached_documents)))
    story.append(_field_table(demographic_rows))

    # --- Tier 1 ---------------------------------------------------------
    story.append(Paragraph("Tier 1 referral triggers", s["h2"]))
    if data.pathognomonic_markers:
        story.append(
            _callout(
                "<b>Tier 1 criteria met.</b> Observed: "
                f"{', '.join(data.pathognomonic_markers)}. In the source cohort these "
                "phenotypes appeared in a majority of TANGO2 patients and in none of "
                "141 controls with other confirmed molecular diagnoses. They are "
                "bedside observations requiring no test. Observing zero events in 141 "
                "controls bounds the false positive rate only loosely; by the rule of "
                "three the one sided 95 percent upper bound is approximately 2.1 "
                "percent. These markers are reported descriptively and are not inputs "
                "to the model below.",
                s["callout"],
                DANGER_SOFT,
                colors.HexColor("#FCA5A5"),
            )
        )
    else:
        story.append(
            Paragraph(
                "Neither paroxysmal lethargy nor compensatory head posture was "
                "recorded. Absence of these markers does not lower the probability "
                "estimate below, which does not use them as inputs.",
                s["body"],
            )
        )

    # --- Prediction -----------------------------------------------------
    story.append(Paragraph("Diagnostic assessment", s["h2"]))
    story.append(
        _field_table(
            [
                ("Model probability", f"{data.probability:.1%}"),
                ("Classification", data.predicted_label),
                ("Confidence band", data.confidence_band),
            ]
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Interpretation", s["h3"]))
    story.append(Paragraph(data.interpretation, s["body"]))

    if data.top_contributors:
        story.append(Paragraph("Phenotypes contributing most to this assessment", s["h3"]))
        rows = [["Baseline phenotype", "Relative contribution"]] + [
            [name, f"{value:.0f}%"] for name, value in data.top_contributors
        ]
        table = Table(rows, colWidths=[4.9 * inch, 2.0 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                    ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
                    ("BACKGROUND", (0, 0), (-1, 0), CANVAS),
                    ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Relative contribution is each phenotype's share of the combined "
                "model weight of every phenotype recorded for this patient, so the "
                "values above sum to 100 percent. No single phenotype drives the "
                "result: in the source study, removing the eight most important "
                "phenotypes left performance essentially unchanged.",
                s["small"],
            )
        )

    # --- Forecast -------------------------------------------------------
    if data.include_forecast:
        story.append(PageBreak())
        story.append(Paragraph("Crisis model output", s["h2"]))
        story.append(
            _callout(
                f"<b>Research use only.</b> {FORECAST_RESEARCH_USE_ONLY}",
                s["callout"],
                DANGER_SOFT,
                colors.HexColor("#FCA5A5"),
            )
        )
        story.append(Spacer(1, 10))
        story.append(
            _field_table(
                [
                    ("Source amplitude", f"{data.beta:.5f}"),
                    ("Critical amplitude", f"{data.beta_critical:.5f}"),
                    ("Regime", (data.regime or "").title()),
                    ("Mean simulated crises", f"{data.mean_crisis_count:.2f} over 900 days"),
                    (
                        "Mean first crossing",
                        "None within horizon"
                        if data.first_crisis_day is None
                        else f"Day {data.first_crisis_day:.0f}",
                    ),
                    (
                        "Realisations with at least one crisis",
                        f"{data.probability_any_crisis:.0%} of {data.n_realizations}",
                    ),
                    ("Random seed", str(data.seed)),
                ]
            )
        )
        if data.regime_note:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Interpretation", s["h3"]))
            story.append(Paragraph(data.regime_note, s["body"]))
        story.append(Paragraph("What this output is and is not", s["h3"]))
        story.append(Paragraph(TREATMENT_NAIVE_NOTE, s["body"]))
        story.append(
            Paragraph(
                "The clinically meaningful quantity is the patient position relative "
                "to the critical amplitude, not any individual simulated day. The "
                "intended use of this model is to graduate monitoring intensity, not "
                "to name a date.",
                s["body"],
            )
        )

    # --- Methodology ----------------------------------------------------
    story.append(Paragraph("Methodology and model performance", s["h2"]))
    story.append(
        _field_table(
            [
                ("Model", "Random Forest, 500 trees, balanced class weights"),
                ("Features", "31 shared vocabulary baseline HPO terms"),
                ("Validation", "Repeated stratified 10 fold cross validation"),
                ("Discrimination", f"AUC {data.auc:.3f}"),
                ("Sensitivity", f"{data.sensitivity:.1%}"),
                ("Specificity", f"{data.specificity:.1%}"),
                ("Calibration", f"Brier score {data.brier:.3f}"),
                ("Model provenance", data.model_provenance.upper()),
                ("Model version", data.model_version),
            ]
        )
    )

    story.append(Paragraph("Scope and limitations", s["h2"]))
    for text in (CLASSIFIER_SCOPE, PERFORMANCE_TRANSFER_CAVEAT):
        story.append(Paragraph(text, s["body"]))
    story.append(
        Paragraph(
            "All crisis defining phenotypes were excluded before the model was fitted, "
            "so no result here can reflect a metabolic crisis that has already "
            "occurred. The analysis was further restricted to phenotype terms used by "
            "both annotating teams, which removes the most plausible route by which a "
            "strong result could be an artefact of annotation style.",
            s["body"],
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Citation", s["h2"]),
                Paragraph(
                    "Tippimath, A., Lalani, S. R., and Liu, Z. A Hybrid Random Forest "
                    "and Reaction Diffusion Framework for Early Identification and "
                    "Crisis Forecasting in TANGO2 Deficiency Disorder. Preprint.",
                    s["small"],
                ),
            ]
        )
    )

    doc.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    return buffer.getvalue()
