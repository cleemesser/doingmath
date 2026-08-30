"""The single source of truth for named colors — replacing the copies scattered across the
GeometricLinearAlgebra and Geometry notebooks. Colors are stored as 0xRRGGBB ints and converted on
demand to whatever a backend wants: a '#rrggbb' string (vedo) or an (r, g, b) float triple (matplotlib).
"""

from __future__ import annotations

# Named palette (kept identical to the values used across the existing notebooks).
BLUE = 0x4FC3F7
ORANGE = 0xFFB74D
GREEN = 0x81C784
RED = 0xEF5350
PURPLE = 0xCE93D8
YELLOW = 0xFFD54F
GREY = 0x888888
FAINT = 0x3A3A4E

# Scene defaults.
BG = 0x0F0F0F
GRID = 0x333333
AXIS = 0x777777


def hexstr(color) -> str:
    """As a '#rrggbb' string (accepts an int or an already-formatted string)."""
    return color if isinstance(color, str) else f"#{int(color):06x}"


def rgb01(color) -> tuple[float, float, float]:
    """As an (r, g, b) triple in [0, 1] (accepts an int or a '#rrggbb' string)."""
    if isinstance(color, str):
        c = int(color.lstrip("#"), 16)
    else:
        c = int(color)
    return ((c >> 16 & 0xFF) / 255.0, (c >> 8 & 0xFF) / 255.0, (c & 0xFF) / 255.0)
