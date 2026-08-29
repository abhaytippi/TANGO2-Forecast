"""Visual language for the application.

Modelled on the visual conventions of clinical software rather than a
marketing site: a fixed navy and slate palette, minimal color used only where
it carries meaning (risk, confirmation, alert), plain typographic hierarchy,
and tables instead of decorative badges wherever a table can carry the same
information.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = ["COLORS", "CUSTOM_CSS", "register_plotly_template"]

COLORS = {
    "ink": "#1A2332",
    "text": "#2D3748",
    "muted": "#718096",
    "line": "#D8DEE6",
    "surface": "#FFFFFF",
    "canvas": "#F4F6F8",
    "header": "#14213D",
    "primary": "#2C5282",
    "primary_dark": "#1A365D",
    "accent": "#0F766E",
    "amber": "#9C6B0B",
    "amber_soft": "#FBF2DE",
    "danger": "#9B2C2C",
    "danger_soft": "#FBEAEA",
    "success": "#276749",
    "success_soft": "#E9F3EC",
    "neutral_soft": "#EEF1F5",
}

CUSTOM_CSS = f"""
<style>
  .stApp {{ background: {COLORS["canvas"]}; }}
  html, body, [class*="css"] {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: {COLORS["text"]};
  }}
  h1, h2, h3, h4 {{ font-weight: 600; color: {COLORS["ink"]}; }}
  h1 {{ font-size: 1.5rem; letter-spacing: 0; }}
  h2 {{ font-size: 1.15rem; margin-top: 1.4rem; border-bottom: 1px solid {COLORS["line"]};
    padding-bottom: .4rem; }}
  h3 {{ font-size: .95rem; color: {COLORS["text"]}; font-weight: 600; }}
  p, li, label, .stMarkdown {{ font-size: .92rem; line-height: 1.55; }}

  /* Top identification bar */
  .app-header {{
    background: {COLORS["header"]}; color: #FFFFFF; padding: .85rem 1.5rem;
    margin: -1rem -1rem 1.1rem -1rem; display: flex; justify-content: space-between;
    align-items: baseline; border-bottom: 3px solid {COLORS["primary"]};
  }}
  .app-header .name {{ font-size: 1.05rem; font-weight: 600; letter-spacing: .01em; }}
  .app-header .tag {{ font-size: .74rem; color: #B7C4D9; letter-spacing: .03em; }}

  .page-title {{ font-size: 1.35rem; font-weight: 650; color: {COLORS["ink"]};
    margin-bottom: .1rem; }}
  .page-sub {{ color: {COLORS["muted"]}; font-size: .88rem; margin-bottom: 1.1rem; }}

  .panel {{
    background: {COLORS["surface"]}; border: 1px solid {COLORS["line"]};
    border-radius: 4px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
  }}
  .panel h4 {{ margin: 0 0 .6rem 0; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .06em; color: {COLORS["muted"]}; font-weight: 700; }}

  .stat {{
    background: {COLORS["surface"]}; border: 1px solid {COLORS["line"]};
    border-left: 3px solid {COLORS["primary"]};
    border-radius: 3px; padding: .85rem 1.1rem; margin-bottom: .7rem;
  }}
  .stat.is-alert {{ border-left-color: {COLORS["danger"]}; }}
  .stat.is-confirm {{ border-left-color: {COLORS["success"]}; }}
  .stat.is-caution {{ border-left-color: {COLORS["amber"]}; }}
  .stat .label {{ font-size: .70rem; text-transform: uppercase; letter-spacing: .06em;
    color: {COLORS["muted"]}; font-weight: 700; margin-bottom: .25rem; }}
  .stat .value {{ font-size: 1.55rem; font-weight: 650; line-height: 1.1; color: {COLORS["ink"]}; }}
  .stat .note {{ color: {COLORS["muted"]}; font-size: .8rem; margin-top: .3rem; }}

  .notice {{
    border-radius: 3px; padding: .7rem 1rem; margin-bottom: .9rem;
    font-size: .86rem; line-height: 1.5; border-left: 3px solid;
  }}
  .notice-alert {{ background: {COLORS["danger_soft"]}; border-color: {COLORS["danger"]};
    color: #5B2020; }}
  .notice-caution {{ background: {COLORS["amber_soft"]}; border-color: {COLORS["amber"]};
    color: #5B4008; }}
  .notice-confirm {{ background: {COLORS["success_soft"]}; border-color: {COLORS["success"]};
    color: #1D4A32; }}
  .notice-neutral {{ background: {COLORS["neutral_soft"]}; border-color: {COLORS["muted"]};
    color: {COLORS["text"]}; }}
  .notice strong {{ font-weight: 700; }}

  .tag {{ display: inline-block; padding: .18rem .55rem; border-radius: 3px;
    font-size: .72rem; font-weight: 700; letter-spacing: .02em; border: 1px solid; }}
  .tag-alert {{ background: {COLORS["danger_soft"]}; color: {COLORS["danger"]};
    border-color: #E2B4B4; }}
  .tag-confirm {{ background: {COLORS["success_soft"]}; color: {COLORS["success"]};
    border-color: #B7D6C1; }}
  .tag-caution {{ background: {COLORS["amber_soft"]}; color: {COLORS["amber"]};
    border-color: #E8CE94; }}
  .tag-neutral {{ background: {COLORS["neutral_soft"]}; color: {COLORS["muted"]};
    border-color: {COLORS["line"]}; }}

  section[data-testid="stSidebar"] {{
    background: {COLORS["surface"]}; border-right: 1px solid {COLORS["line"]};
  }}
  section[data-testid="stSidebar"] .stRadio label {{ font-size: .88rem; }}

  div[data-testid="stMetricValue"] {{ font-size: 1.5rem; color: {COLORS["ink"]}; }}
  div[data-testid="stMetricLabel"] {{ font-size: .74rem; text-transform: uppercase;
    letter-spacing: .05em; color: {COLORS["muted"]}; }}

  .stButton>button, .stDownloadButton>button {{
    border-radius: 3px; border: 1px solid {COLORS["primary"]};
    font-weight: 600; font-size: .86rem;
  }}
  .stButton>button[kind="primary"], .stDownloadButton>button {{
    background: {COLORS["primary"]}; color: white;
  }}

  .sidebar-foot {{ font-size: .74rem; color: {COLORS["muted"]}; line-height: 1.5; }}

  footer, #MainMenu {{ visibility: hidden; }}
</style>
"""


def register_plotly_template() -> None:
    """Register and activate the clinical Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": '"Segoe UI", Arial, sans-serif', "size": 12.5, "color": COLORS["text"]},
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        colorway=[
            COLORS["primary"],
            COLORS["danger"],
            COLORS["accent"],
            COLORS["amber"],
            COLORS["muted"],
        ],
        margin={"l": 55, "r": 25, "t": 45, "b": 50},
        xaxis={
            "gridcolor": COLORS["line"],
            "zerolinecolor": COLORS["line"],
            "linecolor": COLORS["line"],
            "ticks": "outside",
            "tickcolor": COLORS["line"],
        },
        yaxis={
            "gridcolor": COLORS["line"],
            "zerolinecolor": COLORS["line"],
            "linecolor": COLORS["line"],
            "ticks": "outside",
            "tickcolor": COLORS["line"],
        },
        hoverlabel={"bgcolor": COLORS["ink"], "font_size": 11.5},
        legend={
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": COLORS["line"],
            "borderwidth": 1,
        },
    )
    pio.templates["tango2"] = template
    pio.templates.default = "tango2"
