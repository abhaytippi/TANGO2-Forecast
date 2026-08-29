"""Interactive Plotly figures.

Framework-independent builders: each returns a bare ``go.Figure`` so the same
code serves the dashboard, notebooks, and exported reports. Nothing here imports
Streamlit.

The figures correspond to the paper's Figures 2, 3, 5, 6, 7 and 8.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..forecast.solver import ForecastResult
from ..ml.classifier import EvaluationReport
from .theme import COLORS

__all__ = [
    "bifurcation_figure",
    "calibration_figure",
    "ensemble_figure",
    "importance_figure",
    "probability_gauge",
    "roc_figure",
    "spacetime_figure",
    "trajectory_figure",
]


def trajectory_figure(result: ForecastResult, realization: int = 0) -> go.Figure:
    """Single stochastic trajectory with crisis markers (paper, Figure 5)."""
    params = result.parameters
    peak = result.peak_risk[:, realization]
    fig = go.Figure()

    fig.add_hrect(y0=params.r_star, y1=1.0, fillcolor=COLORS["danger"], opacity=0.06, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=peak,
            mode="lines",
            name="Peak risk",
            line={"color": COLORS["primary"], "width": 2},
            hovertemplate="Day %{x:.0f}<br>Peak risk %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=params.r_star,
        line={"color": COLORS["amber"], "dash": "dash", "width": 2},
        annotation_text=f"Crisis threshold r* = {params.r_star}",
        annotation_position="top right",
    )

    days = result.crisis_days[realization]
    if days.size:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=np.full(days.size, params.r_star),
                mode="markers",
                name="Threshold crossing",
                marker={
                    "symbol": "star",
                    "size": 15,
                    "color": COLORS["danger"],
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="Simulated crossing, day %{x:.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Simulated risk trajectory",
        xaxis_title="Days since baseline assessment",
        yaxis_title="Peak risk across subsystems",
        yaxis_range=[0, 1],
        height=420,
        hovermode="x unified",
    )
    return fig


def ensemble_figure(result: ForecastResult, n_paths: int = 4) -> go.Figure:
    """Ensemble with percentile bands (paper, Figure 6).

    Uncertainty widening with horizon is the point of this figure, and the
    reason the interface never presents a single predicted date.
    """
    _, values = result.peak_risk_quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    times = result.times
    fig = go.Figure()

    for lower, upper, opacity, label in (
        (0, 4, 0.12, "5th-95th percentile"),
        (1, 3, 0.22, "25th-75th percentile"),
    ):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([values[upper], values[lower][::-1]]),
                fill="toself",
                fillcolor=f"rgba(29, 78, 216, {opacity})",
                line={"width": 0},
                name=label,
                hoverinfo="skip",
            )
        )

    for index in range(min(n_paths, result.n_realizations)):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=result.peak_risk[:, index],
                mode="lines",
                line={"width": 1, "color": COLORS["muted"]},
                opacity=0.45,
                name="Individual realisation" if index == 0 else None,
                showlegend=index == 0,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=values[2],
            mode="lines",
            name="Median",
            line={"color": COLORS["primary"], "width": 2.5},
            hovertemplate="Day %{x:.0f}<br>Median peak risk %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=result.parameters.r_star,
        line={"color": COLORS["amber"], "dash": "dash", "width": 2},
        annotation_text="Crisis threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Forecast ensemble, {result.n_realizations} realisations",
        xaxis_title="Days since baseline assessment",
        yaxis_title="Peak risk across subsystems",
        yaxis_range=[0, 1],
        height=440,
        hovermode="x unified",
    )
    return fig


def spacetime_figure(result: ForecastResult) -> go.Figure:
    """Space-time evolution of the risk field (paper, Figure 7)."""
    if result.field is None or result.field_times is None:
        raise ValueError("run the solver with store_field=True to draw this figure")

    fig = go.Figure(
        go.Heatmap(
            z=result.field.T,
            x=result.field_times,
            y=result.x,
            colorscale="RdYlBu_r",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "Risk r(x,t)"},
            hovertemplate="Day %{x:.0f}<br>Subsystem %{y:.2f}<br>Risk %{z:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=result.parameters.x0,
        line={"color": COLORS["ink"], "dash": "dot", "width": 1.5},
        annotation_text=f"Vulnerability locus x0 = {result.parameters.x0}",
    )
    fig.update_layout(
        title="Where risk concentrates and how it spreads",
        xaxis_title="Days since baseline assessment",
        yaxis_title="Latent biomarker coordinate",
        height=420,
    )
    return fig


def bifurcation_figure(
    betas: np.ndarray,
    peaks: np.ndarray,
    beta_critical: float,
    r_star: float,
    patient_beta: float | None = None,
) -> go.Figure:
    """Steady-state peak risk against source amplitude (paper, Figure 8a)."""
    fig = go.Figure()
    fig.add_vrect(
        x0=float(betas.min()),
        x1=beta_critical,
        fillcolor=COLORS["success"],
        opacity=0.07,
        line_width=0,
        annotation_text="Quiescent regime",
        annotation_position="top left",
    )
    fig.add_vrect(
        x0=beta_critical,
        x1=float(betas.max()),
        fillcolor=COLORS["danger"],
        opacity=0.07,
        line_width=0,
        annotation_text="Recurrent crises",
        annotation_position="top right",
    )

    fig.add_trace(
        go.Scatter(
            x=betas,
            y=peaks,
            mode="lines",
            name="Steady-state peak risk",
            line={"color": COLORS["primary"], "width": 2.5},
            hovertemplate="beta %{x:.5f}<br>Peak risk %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=r_star, line={"color": COLORS["amber"], "dash": "dash"}, annotation_text=f"r* = {r_star}"
    )
    fig.add_vline(
        x=beta_critical,
        line={"color": COLORS["danger"], "dash": "dot"},
        annotation_text=f"beta* = {beta_critical:.5f}",
    )

    if patient_beta is not None:
        fig.add_trace(
            go.Scatter(
                x=[patient_beta],
                y=[float(np.interp(patient_beta, betas, peaks))],
                mode="markers",
                name="This patient",
                marker={
                    "size": 15,
                    "color": COLORS["ink"],
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "white"},
                },
            )
        )

    fig.update_layout(
        title="The bifurcation: where a patient sits decides the regime",
        xaxis_title="Source amplitude beta",
        yaxis_title="Steady-state peak risk",
        height=420,
    )
    return fig


def roc_figure(report: EvaluationReport) -> go.Figure:
    """Cross-validated ROC curve (paper, Figure 2a)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=report.roc_fpr,
            y=report.roc_tpr,
            mode="lines",
            name=f"Random Forest (AUC = {report.auc:.3f})",
            fill="tozeroy",
            fillcolor="rgba(29, 78, 216, 0.08)",
            line={"color": COLORS["primary"], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance (AUC = 0.500)",
            line={"color": COLORS["muted"], "dash": "dash", "width": 1.5},
        )
    )
    fig.update_layout(
        title="Discrimination",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=400,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        legend={"x": 0.45, "y": 0.08},
    )
    return fig


def calibration_figure(report: EvaluationReport) -> go.Figure:
    """Calibration curve (paper, Figure 3a).

    Discrimination alone is insufficient if a probability is to inform a
    referral: the question is whether a stated 70% actually means 70%.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": COLORS["muted"], "dash": "dash", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=report.calibration_predicted,
            y=report.calibration_observed,
            mode="lines+markers",
            name="This model",
            line={"color": COLORS["primary"], "width": 2.5},
            marker={"size": 9},
        )
    )
    fig.update_layout(
        title=f"Calibration (Brier {report.brier:.3f}, skill {report.brier_skill:.2f})",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed fraction with TANGO2",
        height=400,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig


def importance_figure(names: list[str], values: list[float], top_n: int = 12) -> go.Figure:
    """Feature importance (paper, Figure 4a).

    Correlated phenotypes share credit, so importances identify informative
    clusters rather than independent causes.
    """
    order = np.argsort(values)[-top_n:]
    fig = go.Figure(
        go.Bar(
            x=[values[i] for i in order],
            y=[names[i] for i in order],
            orientation="h",
            marker_color=COLORS["primary"],
            hovertemplate="%{y}<br>Importance %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Which baseline phenotypes the model relies on",
        xaxis_title="Gini importance",
        yaxis_title=None,
        height=max(360, 26 * top_n),
    )
    return fig


def probability_gauge(probability: float) -> go.Figure:
    """Prediction gauge, banded so the ambiguous middle reads as ambiguous."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": COLORS["ink"], "thickness": 0.25},
                "steps": [
                    {"range": [0, 20], "color": COLORS["success_soft"]},
                    {"range": [20, 50], "color": "#F1F5F9"},
                    {"range": [50, 80], "color": COLORS["amber_soft"]},
                    {"range": [80, 100], "color": COLORS["danger_soft"]},
                ],
                "threshold": {
                    "line": {"color": COLORS["danger"], "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=250, margin={"l": 30, "r": 30, "t": 30, "b": 10})
    return fig
