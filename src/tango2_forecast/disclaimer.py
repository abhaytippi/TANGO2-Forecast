"""Single source of truth for the scope-of-use statements.

Every surface of this platform -- library docstrings, dashboard pages, exported
PDFs and CSVs -- draws its scope language from here, so that the boundary of the
underlying research cannot drift between them. If a claim is weakened or
strengthened in the paper, it is changed once, in this module.

The distinction the paper draws in Section 4 and Section 9 is the one that
matters and is preserved verbatim in spirit: the reaction--diffusion model is
*mathematically verified* (the equations are solved correctly) and *not
biologically validated* (no forecast has been compared against an observed
crisis, because the source dataset contains no crisis dates).
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CLASSIFIER_SCOPE",
    "FORECAST_RESEARCH_USE_ONLY",
    "NOT_A_MEDICAL_DEVICE",
    "PERFORMANCE_TRANSFER_CAVEAT",
    "SYNTHETIC_DATA_WARNING",
]

NOT_A_MEDICAL_DEVICE: Final[str] = (
    "TANGO2-Forecast is research software. It is not a medical device, has not "
    "been reviewed or cleared by any regulator, and must not be used as the sole "
    "basis for any diagnostic or treatment decision. All clinical decisions "
    "remain the responsibility of a qualified clinician."
)

FORECAST_RESEARCH_USE_ONLY: Final[str] = (
    "RESEARCH USE ONLY. The reaction-diffusion crisis model is hypothesis-"
    "generating. Its solver has been verified for convergence, conservation and "
    "stability, but the model itself has never been validated against observed "
    "crisis dates: the source cohort contained none, and its parameters were "
    "calibrated to reproduce qualitative published behaviour rather than fitted "
    "to outcomes. Simulated crisis days are model outputs, not predictions with "
    "demonstrated accuracy. Treating any specific predicted day as a clinical "
    "decision point would misuse the model."
)

CLASSIFIER_SCOPE: Final[str] = (
    "The diagnostic classifier was evaluated by repeated stratified "
    "cross-validation on 90 molecularly confirmed TANGO2 patients and 141 "
    "Undiagnosed Diseases Network controls. It is intended to support a referral "
    "decision -- whether to send a child for TANGO2 sequencing -- and not to "
    "confirm or exclude a diagnosis, which requires molecular testing."
)

PERFORMANCE_TRANSFER_CAVEAT: Final[str] = (
    "Reported performance comes from specialist-referred, deeply phenotyped "
    "cohorts. In an unselected developmental clinic the prevalence of TANGO2 "
    "deficiency is far lower and phenotyping far less complete, so both positive "
    "predictive value and discrimination will be lower than the published "
    "figures. Prospective testing on undiagnosed patients is the only thing that "
    "would settle this."
)

SYNTHETIC_DATA_WARNING: Final[str] = (
    "SYNTHETIC DATA. This model was fitted to a simulated cohort generated from "
    "published summary statistics, not to patient records. The source cohorts are "
    "not publicly redistributable. Any performance figure produced from synthetic "
    "data describes the simulation and says nothing about real-world accuracy."
)

# Treatment context that the crisis model deliberately omits (paper, Section 9).
TREATMENT_NAIVE_NOTE: Final[str] = (
    "The crisis model is treatment-naive and trigger-naive. It omits B-complex "
    "supplementation, which substantially reduces crisis frequency, so it "
    "describes an untreated baseline and will over-predict crises in adequately "
    "supplemented patients. Specific triggers such as fasting or febrile illness "
    "are folded into a generic noise term rather than represented explicitly."
)
