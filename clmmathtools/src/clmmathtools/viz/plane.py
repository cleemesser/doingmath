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

    def polygon(
        self, pts, facecolor=palette.BLUE, alpha=0.3, edgecolor=None, edgewidth=2.0
    ):
        """A filled polygon (e.g. a parallelogram for a signed area)."""
        self.primitives.append(P.Polygon(pts, facecolor, alpha, edgecolor, edgewidth))
        return self

    def parallelogram(
        self,
        origin,
        u,
        v,
        facecolor=palette.BLUE,
        alpha=0.3,
        edgecolor=None,
        edgewidth=2.0,
    ):
        """The filled parallelogram spanned by ``u`` and ``v`` at ``origin`` (area = |u ∧ v|)."""
        o, u, v = (np.asarray(x, float).reshape(2) for x in (origin, u, v))
        return self.polygon(
            [o, o + u, o + u + v, o + v], facecolor, alpha, edgecolor, edgewidth
        )

    # ── maps: push the plane through a transform (see maps.py) ─
    def apply_matrix(self, M, **kw):
        """Warp the coordinate grid by a 2×2 matrix and draw its column images."""
        from . import maps

        return maps.apply_matrix(self, M, **kw)

    def apply_complex(self, g, **kw):
        """Draw the conformal image of the grid under a complex function g: ℂ→ℂ."""
        from . import maps

        return maps.apply_complex(self, g, **kw)

    def show_operator(self, f, **kw):
        """Overlay a faint domain grid + probe shape with their bold image under ``f``."""
        from . import maps

        return maps.show_operator(self, f, **kw)

    def push(self, f, **kw):
        """Draw the image of the domain under any map (2×2 matrix or point-map)."""
        from . import maps

        return maps.push(self, f, **kw)

    def field(self, f, **kw):
        """Draw a 2D vector field ``f`` (a 2×2 matrix or a point-map) as a grid of arrows."""
        from . import maps

        return maps.vector_field(self, f, **kw)

    def field_complex(self, g, **kw):
        """Draw a complex vector field g: ℂ→ℂ (arrow = (Re g, Im g)) as a grid of arrows."""
        from . import maps

        return maps.vector_field(self, maps.from_complex(g), **kw)

    def phase_portrait(self, f, *, res=None, scheme="enhanced", alpha=1.0, **kw):
        """Fill the plane with the Wegert phase portrait of a complex function f: ℂ→ℂ.

        ``res`` defaults to the view's pixel width: a raster coarser than the pixels it is stretched
        across shows blocky contours no anti-aliasing can recover. Pass ``res`` explicitly to trade
        sharpness for speed (animation frames), or to oversample.
        """
        from . import phase

        if res is None:
            res = max(self.view.size)
        rgb = phase.phase_portrait(
            f, extent=self.view.extent, res=res, scheme=scheme, **kw
        )
        return self.raster(rgb, alpha=alpha)

    # ── output ───────────────────────────────────────────────
    def _scene(self):
        return P.Scene(self.view, list(self.primitives))

    def display(self, *, format=None, interactive=None, vedo_display=None):
        """Render inline. ``interactive=True`` (vedo backend) gives a live, orbitable widget.

        ``format`` picks the inline image type; the matplotlib backend defaults to "svg".
        Pass ``format="png"`` for raster-heavy scenes (phase portraits, ``raster``), where
        SVG merely wraps an embedded PNG and ends up larger.
        """
        return get_backend(self._backend).render(
            self._scene(),
            format=format,
            interactive=interactive,
            vedo_display=vedo_display,
        )

    def save(self, path, *, format=None):
        """Write the scene to ``path`` (a filename or an open buffer).

        ``format`` ("png", "svg", ...) is required for buffers, which have no suffix to infer from.
        SVG is the better choice for a scene that re-renders on every widget tick: it embeds
        straight into the DOM, so there is no image asset to re-fetch and re-decode per frame.
        """
        return get_backend(self._backend).render(
            self._scene(), save=path, format=format
        )
