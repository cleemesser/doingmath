"""mathviz — plane & complex-function visualization over pluggable matplotlib/vedo backends.

Draw on the 2D / complex plane, push it through maps (matrices, operators, complex functions), and
color it as a Wegert phase portrait. Factors out the duplicated plane-viz helpers from the
GeometricLinearAlgebra and Geometry notebooks.
"""

from __future__ import annotations

from . import palette, primitives
from .backends import get_backend, set_backend
from .palette import BG, BLUE, FAINT, GREEN, GREY, ORANGE, PURPLE, RED, YELLOW
from .plane import Plane

__all__ = [
    "Plane",
    "set_backend",
    "get_backend",
    "palette",
    "primitives",
    "BLUE",
    "ORANGE",
    "GREEN",
    "RED",
    "PURPLE",
    "YELLOW",
    "GREY",
    "FAINT",
    "BG",
]

__version__ = "0.1.0"
