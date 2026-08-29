"""TANGO2-Forecast -- open clinical decision support platform.

A translation of the hybrid Random Forest / reaction--diffusion framework of

    Tippimath, A., Lalani, S. R., & Liu, Z. *A Hybrid Random Forest and
    Reaction--Diffusion Framework for Early Identification and Crisis
    Forecasting in TANGO2 Deficiency Disorder.* Preprint.

into installable, tested research software.

This package is **not a medical device**; see :mod:`tango2_forecast.disclaimer`.
"""

from .disclaimer import (
    CLASSIFIER_SCOPE,
    FORECAST_RESEARCH_USE_ONLY,
    NOT_A_MEDICAL_DEVICE,
)

__version__ = "0.1.0"

__all__ = [
    "CLASSIFIER_SCOPE",
    "FORECAST_RESEARCH_USE_ONLY",
    "NOT_A_MEDICAL_DEVICE",
    "__version__",
]
