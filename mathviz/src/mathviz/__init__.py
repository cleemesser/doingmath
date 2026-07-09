"""mathviz — plane & complex-function visualization over pluggable matplotlib/vedo backends.

Draw on the 2D / complex plane, push it through maps (matrices, operators, complex functions), and
color it as a Wegert phase portrait. Factors out the duplicated plane-viz helpers from the
GeometricLinearAlgebra and Geometry notebooks.
"""

from __future__ import annotations

from . import animate, maps, palette, phase, primitives
from .animate import (
    animate_matrix,
    homotopy,
    matrix_path,
    mobius,
    mobius_path,
    scrubber,
    scrubber2,
    to_gif,
)
from .backends import (
    get_backend,
    in_notebook,
    set_backend,
    set_interactive,
    set_vedo_display,
)
from .maps import FLAG, UNIT_SQUARE, unit_circle
from .notebook import show_expr
from .palette import BG, BLUE, FAINT, GREEN, GREY, ORANGE, PURPLE, RED, YELLOW
from .phase import phase_portrait
from .plane import Plane
from .space3d import Space3D

__all__ = [
    "Plane",
    "Space3D",
    "set_backend",
    "get_backend",
    "set_interactive",
    "set_vedo_display",
    "in_notebook",
    "show_expr",
    "animate",
    "animate_matrix",
    "matrix_path",
    "scrubber",
    "scrubber2",
    "to_gif",
    "homotopy",
    "mobius",
    "mobius_path",
    "maps",
    "phase",
    "phase_portrait",
    "palette",
    "primitives",
    "FLAG",
    "UNIT_SQUARE",
    "unit_circle",
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
