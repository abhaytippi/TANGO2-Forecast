"""Dashboard components: theme and interactive figures."""

from . import figures
from .theme import COLORS, CUSTOM_CSS, register_plotly_template

__all__ = ["COLORS", "CUSTOM_CSS", "figures", "register_plotly_template"]
