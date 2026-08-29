"""Mathematical crisis-forecast engine (Module A).

Implements the stochastic reaction--diffusion model of Tippimath, Lalani & Liu,
including the IMEX solver of Algorithm 1, the steady-state analysis that yields
the critical source amplitude ``beta*``, the severity mapping of Eq. (8), and
the numerical verification suite of Section 4.
"""

from .operators import (
    TridiagonalOperator,
    amplification_factor,
    build_implicit_operator,
    laplacian_bands,
    thomas_solve,
)
from .parameters import (
    BETA_CRITICAL_PUBLISHED,
    BETA_MAX,
    BETA_MIN,
    CrisisModelParameters,
    GridSpec,
)
from .sensitivity import (
    SENSITIVITY_PARAMETERS,
    ParameterSensitivity,
    sensitivity_analysis,
)
from .severity import RegimeAssessment, SeverityScale, classify_regime
from .solver import ForecastResult, IMEXCrisisSolver
from .steady_state import (
    SteadyState,
    bifurcation_curve,
    critical_amplitude,
    steady_state,
)
from .verification import (
    RefinementLevel,
    mass_conservation_drift,
    max_amplification,
    spatial_convergence,
    temporal_convergence,
)

__all__ = [
    "BETA_CRITICAL_PUBLISHED",
    "BETA_MAX",
    "BETA_MIN",
    "SENSITIVITY_PARAMETERS",
    "CrisisModelParameters",
    "ForecastResult",
    "GridSpec",
    "IMEXCrisisSolver",
    "ParameterSensitivity",
    "RefinementLevel",
    "RegimeAssessment",
    "SeverityScale",
    "SteadyState",
    "TridiagonalOperator",
    "amplification_factor",
    "bifurcation_curve",
    "build_implicit_operator",
    "classify_regime",
    "critical_amplitude",
    "laplacian_bands",
    "mass_conservation_drift",
    "max_amplification",
    "sensitivity_analysis",
    "spatial_convergence",
    "steady_state",
    "temporal_convergence",
    "thomas_solve",
]
