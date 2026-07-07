"""The ``Plane`` facade — a top-down view of the x–y (complex) plane you draw on with chainable
calls. It accumulates backend-neutral primitives and hands them to a backend to render. This is the
single API that subsumes the ``Plane2D`` / ``new_plot`` helpers previously copied across the
GeometricLinearAlgebra and Geometry notebooks.
"""

from __future__ import annotations

import numpy as np

from . import palette, primitives as P
from .backends import get_backend


class Plane:
    def __init__(
        self,
        extent=3.0,
        *,
        backend=None,
        bg=palette.BG,
        size=(640, 640),
        grid=True,
        axes=True,
    ):
        self.view = P.View(extent=float(extent), bg=bg, size=size)
        self.primitives: list = []
        self._backend = backend  # name or None (→ global default at render time)
        if grid:
            self.grid()
        if axes:
            self.axes()

    # ── structure ────────────────────────────────────────────
    def grid(self, step=1.0, color=palette.GRID, alpha=0.9, width=1.0):
        self.primitives.append(P.Grid(self.view.extent, step, color, alpha, width))
        return self

    def axes(self, color=palette.AXIS, width=1.5, alpha=1.0):
        E = self.view.extent
        self.primitives.append(P.Segment([-E, 0], [E, 0], color, width, alpha))
        self.primitives.append(P.Segment([0, -E], [0, E], color, width, alpha))
        return self

    # ── primitives ───────────────────────────────────────────
    def vector(
        self, vec, origin=(0, 0), color=palette.RED, label=None, width=2.5, alpha=1.0
    ):
        origin = np.asarray(origin, float).reshape(2)
        head = origin + np.asarray(vec, float).reshape(2)
        self.primitives.append(P.Arrow(origin, head, color, width, alpha, label))
        return self

    def basis(self, colors=(palette.RED, palette.GREEN), labels=("e1", "e2")):
        self.vector([1, 0], color=colors[0], label=labels[0])
        self.vector([0, 1], color=colors[1], label=labels[1])
        return self

    def segment(self, p, q, color=palette.GREY, width=2.0, alpha=1.0):
        self.primitives.append(P.Segment(p, q, color, width, alpha))
        return self

    def line(self, point, direction, color=palette.PURPLE, width=3.0, alpha=0.9):
        """A full line through ``point`` with direction ``direction``, clipped to the view."""
        d = np.asarray(direction, float).reshape(2)
        d = d / np.linalg.norm(d)
        p = np.asarray(point, float).reshape(2)
        t = 2 * self.view.extent
        return self.segment(p - t * d, p + t * d, color, width, alpha)

    def curve(self, pts, color=palette.BLUE, width=3.0, alpha=1.0, closed=False):
        self.primitives.append(P.Polyline(pts, color, width, alpha, closed))
        return self

    def points(self, pts, color=palette.ORANGE, size=8.0, alpha=1.0):
        self.primitives.append(P.Points(pts, color, size, alpha))
        return self

    def text(self, pos, s, color=palette.GREY, size=12.0):
        self.primitives.append(P.Text(pos, s, color, size))
        return self

    def raster(self, rgb, extent=None, alpha=1.0):
        """Fill the plane with an RGB image (the substrate for phase portraits)."""
        if extent is None:
            E = self.view.extent
            extent = (-E, E, -E, E)
        self.primitives.append(P.Raster(np.asarray(rgb), tuple(extent), alpha))
        return self

    # ── output ───────────────────────────────────────────────
    def _scene(self):
        return P.Scene(self.view, list(self.primitives))

    def display(self):
        return get_backend(self._backend).render(self._scene())

    def save(self, path):
        return get_backend(self._backend).render(self._scene(), save=path)
