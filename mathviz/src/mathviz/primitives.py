"""Backend-neutral scene primitives.

A ``Plane`` records a ``View`` plus a list of these dataclasses; a backend walks the list and draws
each one. Keeping the scene as plain, inspectable data (rather than immediate backend calls) is what
makes the backends pluggable and the geometry unit-testable without a display.

Coordinates are plane coordinates (x, y); points are given as array-likes and normalized to float
NumPy arrays of shape (2,) or (N, 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import palette


def _pt(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(2)


def _pts(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(-1, 2)


@dataclass
class View:
    """The camera/frame: a square [-extent, extent]² top-down window with fixed scale."""

    extent: float = 3.0
    bg: int = palette.BG
    size: tuple[int, int] = (640, 640)


@dataclass
class Grid:
    extent: float
    step: float = 1.0
    color: int = palette.GRID
    alpha: float = 1.0
    width: float = 1.0


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    color: int = palette.GREY
    width: float = 2.0
    alpha: float = 1.0

    def __post_init__(self):
        self.p0, self.p1 = _pt(self.p0), _pt(self.p1)


@dataclass
class Polyline:
    pts: np.ndarray
    color: int = palette.BLUE
    width: float = 3.0
    alpha: float = 1.0
    closed: bool = False

    def __post_init__(self):
        self.pts = _pts(self.pts)


@dataclass
class Arrow:
    tail: np.ndarray
    head: np.ndarray
    color: int = palette.RED
    width: float = 2.5
    alpha: float = 1.0
    label: str | None = None

    def __post_init__(self):
        self.tail, self.head = _pt(self.tail), _pt(self.head)


@dataclass
class Points:
    pts: np.ndarray
    color: int = palette.ORANGE
    size: float = 8.0
    alpha: float = 1.0

    def __post_init__(self):
        self.pts = _pts(self.pts)


@dataclass
class Text:
    pos: np.ndarray
    text: str
    color: int = palette.GREY
    size: float = 12.0

    def __post_init__(self):
        self.pos = _pt(self.pos)


@dataclass
class Raster:
    """An RGB image filling a rectangle of the plane — the substrate for phase portraits.

    rgb: (H, W, 3) array, float in [0, 1] or uint8. extent: (xmin, xmax, ymin, ymax).
    """

    rgb: np.ndarray
    extent: tuple[float, float, float, float]
    alpha: float = 1.0


@dataclass
class Polygon:
    """A filled 2D polygon (e.g. a parallelogram / area). ``edgecolor=None`` → no outline."""

    pts: np.ndarray
    facecolor: int = palette.BLUE
    alpha: float = 0.3
    edgecolor: int | None = None
    edgewidth: float = 2.0

    def __post_init__(self):
        self.pts = _pts(self.pts)


# ── 3D primitives (rendered by a Space3D scene) ───────────────────────────────────
@dataclass
class View3D:
    """A perspective 3D frame, roughly bounding [-bounds, bounds]³, viewed from (elev, azim).

    ``up`` names the world axis drawn vertically: ``"z"`` (the matplotlib/analytic-landscape
    convention — height is up) or ``"y"`` (VTK's default, which lays z out to the side).
    """

    bounds: float = 3.0
    bg: int = palette.BG
    size: tuple[int, int] = (720, 720)
    elev: float = 22.0
    azim: float = -60.0
    up: str = "z"


@dataclass
class Arrow3D:
    tail: np.ndarray
    head: np.ndarray
    color: int = palette.RED
    width: float = 2.5
    alpha: float = 1.0

    def __post_init__(self):
        self.tail = np.asarray(self.tail, float).reshape(3)
        self.head = np.asarray(self.head, float).reshape(3)


@dataclass
class Line3D:
    pts: np.ndarray
    color: int = palette.BLUE
    width: float = 3.0
    alpha: float = 1.0

    def __post_init__(self):
        self.pts = np.asarray(self.pts, float).reshape(-1, 3)


@dataclass
class Points3D:
    pts: np.ndarray
    color: int = palette.ORANGE
    size: float = 8.0
    alpha: float = 1.0

    def __post_init__(self):
        self.pts = np.asarray(self.pts, float).reshape(-1, 3)


@dataclass
class Surface:
    """A gridded surface: X, Y, Z are (m, n) arrays; optional per-vertex ``colors`` (m, n, 3)."""

    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    colors: np.ndarray | None = None
    color: int = palette.BLUE
    alpha: float = 1.0
    wireframe: bool = False


@dataclass
class Mesh3D:
    """A filled 3D mesh: ``verts`` (N, 3) and ``faces`` (list of index lists, any polygon degree).

    Builds solids (parallelepipeds) and flat faces (a bivector parallelogram). ``edgecolor=None`` →
    no wireframe; optional per-face ``colors`` (F, 3) overrides the flat ``facecolor``.
    """

    verts: np.ndarray
    faces: list
    facecolor: int = palette.BLUE
    alpha: float = 0.5
    colors: np.ndarray | None = None
    edgecolor: int | None = None
    edgewidth: float = 1.0

    def __post_init__(self):
        self.verts = np.asarray(self.verts, float).reshape(-1, 3)


@dataclass
class Scene:
    view: View
    primitives: list = field(default_factory=list)
