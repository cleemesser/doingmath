"""``Space3D`` — a perspective 3D scene, the 3D sibling of ``Plane``.

Same idea as ``Plane``: chainable calls record backend-neutral 3D primitives (``Arrow3D``, ``Line3D``,
``Points3D``, ``Surface``); a backend renders them. Two headline builders:

* :meth:`Space3D.field` — a **3D vector field** ``f: ℝ³→ℝ³`` as a lattice of arrows (uniform length,
  magnitude by color, matching the 2D convention).
* :meth:`Space3D.landscape` — the **analytic landscape** of a complex ``f``: the surface ``|f(z)|``
  colored by ``arg f(z)`` (reusing :func:`mathviz.phase.colorize`), the 3D form of a phase portrait.
"""

from __future__ import annotations

import numpy as np

from . import palette, primitives as P
from .backends import get_backend


def _as_field3(f):
    """Accept a 3×3 matrix (linear field x ↦ Mx) or a callable ``(N,3)->(N,3)``."""
    if isinstance(f, np.ndarray) or np.ndim(f) == 2:
        M = np.asarray(f, float).reshape(3, 3)
        return lambda pts: np.asarray(pts, float).reshape(-1, 3) @ M.T
    return f


def _scheme_colors(w, scheme):
    """Phase-portrait colors for a complex array, selecting contours from a scheme name."""
    from .phase import colorize

    return colorize(
        w,
        phase_contours=(scheme in ("phase", "enhanced")),
        modulus_contours=(scheme in ("modulus", "enhanced")),
    )


def _cmap_colors(mag, cmap):
    import matplotlib as mpl
    import matplotlib.cm as cm

    fn = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else mpl.colormaps[cmap]
    lo, hi = float(np.nanmin(mag)), float(np.nanmax(mag))
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
    rgb = (np.asarray(fn(norm(mag)))[:, :3] * 255).astype(int)
    return [(int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in rgb]


class Space3D:
    def __init__(
        self,
        bounds=3.0,
        *,
        backend="vedo",
        bg=palette.BG,
        size=(720, 720),
        elev=22.0,
        azim=-60.0,
    ):
        self.view = P.View3D(
            bounds=float(bounds), bg=bg, size=size, elev=elev, azim=azim
        )
        self.primitives: list = []
        self._backend = backend

    # ── primitives ───────────────────────────────────────────
    def arrow(self, tail, head, color=palette.RED, width=2.5, alpha=1.0):
        self.primitives.append(P.Arrow3D(tail, head, color, width, alpha))
        return self

    def line(self, pts, color=palette.BLUE, width=3.0, alpha=1.0):
        self.primitives.append(P.Line3D(pts, color, width, alpha))
        return self

    def points(self, pts, color=palette.ORANGE, size=8.0, alpha=1.0):
        self.primitives.append(P.Points3D(pts, color, size, alpha))
        return self

    def surface(
        self, X, Y, Z, colors=None, color=palette.BLUE, alpha=1.0, wireframe=False
    ):
        self.primitives.append(
            P.Surface(
                np.asarray(X),
                np.asarray(Y),
                np.asarray(Z),
                colors,
                color,
                alpha,
                wireframe,
            )
        )
        return self

    def mesh(
        self,
        verts,
        faces,
        facecolor=palette.BLUE,
        alpha=0.5,
        colors=None,
        edgecolor=None,
        edgewidth=1.0,
    ):
        """A filled 3D mesh from ``verts`` (N,3) and ``faces`` (list of index lists)."""
        self.primitives.append(
            P.Mesh3D(verts, list(faces), facecolor, alpha, colors, edgecolor, edgewidth)
        )
        return self

    def polygon(
        self, verts, facecolor=palette.BLUE, alpha=0.5, edgecolor=None, edgewidth=1.0
    ):
        """A single filled flat polygon face in 3D (e.g. a bivector parallelogram)."""
        verts = np.asarray(verts, float).reshape(-1, 3)
        return self.mesh(
            verts,
            [list(range(len(verts)))],
            facecolor,
            alpha,
            None,
            edgecolor,
            edgewidth,
        )

    def parallelogram(
        self,
        origin,
        u,
        v,
        facecolor=palette.BLUE,
        alpha=0.5,
        edgecolor=None,
        edgewidth=1.0,
    ):
        """The filled 3D parallelogram spanned by ``u`` and ``v`` at ``origin`` (a bivector)."""
        o, u, v = (np.asarray(x, float).reshape(3) for x in (origin, u, v))
        return self.polygon(
            [o, o + u, o + u + v, o + v], facecolor, alpha, edgecolor, edgewidth
        )

    def parallelepiped(
        self,
        origin,
        a,
        b,
        c,
        facecolor=palette.BLUE,
        alpha=0.35,
        edgecolor=palette.GREY,
        edgewidth=1.5,
    ):
        """The filled parallelepiped spanned by ``a, b, c`` at ``origin`` (volume = |det[a b c]|)."""
        o, a, b, c = (np.asarray(x, float).reshape(3) for x in (origin, a, b, c))
        verts = [o, o + a, o + a + b, o + b, o + c, o + a + c, o + a + b + c, o + b + c]
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        return self.mesh(verts, faces, facecolor, alpha, None, edgecolor, edgewidth)

    # ── 3D vector field ──────────────────────────────────────
    def field(
        self,
        f,
        *,
        n=7,
        color=palette.BLUE,
        width=2.0,
        scale=None,
        normalize=True,
        cmap="viridis",
    ):
        """Draw a 3D vector field ``f`` (a 3×3 matrix or ``(N,3)->(N,3)``) as a lattice of arrows.

        Like the 2D :meth:`Plane.field`, arrows default to **uniform length with magnitude by color**.
        """
        fmap = _as_field3(f)
        B = self.view.bounds
        xs = np.linspace(-B, B, n)
        pts = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
        vec = np.asarray(fmap(pts), float).reshape(-1, 3)
        mag = np.linalg.norm(vec, axis=1)
        ok = np.isfinite(mag) & (mag > 0)
        cell = (2 * B) / (n - 1)

        if normalize:
            disp = np.zeros_like(vec)
            disp[ok] = vec[ok] / mag[ok, None] * (cell * 0.4)
        else:
            peak = np.nanmax(mag[ok]) if ok.any() else 1.0
            disp = vec * (scale if scale is not None else (cell * 0.85) / peak)

        cols = _cmap_colors(mag, cmap) if cmap is not None else None
        for i in range(len(pts)):
            if not ok[i]:
                continue
            self.arrow(
                pts[i],
                pts[i] + disp[i],
                color=(cols[i] if cols else color),
                width=width,
            )
        return self

    # ── analytic landscape (3D phase portrait) ───────────────
    def landscape(
        self, f, *, extent=None, res=140, scheme="enhanced", zmax=None, log=False, **kw
    ):
        """Surface ``|f(z)|`` over the complex plane, colored by ``arg f(z)`` — a 3D phase portrait.

        ``extent`` defaults to the scene ``bounds``; the height is clipped to ``zmax`` (default =
        ``bounds``), or set ``log=True`` to plot ``log(1+|f|)`` when the modulus range is large.
        """
        from .phase import colorize, domain

        E = self.view.bounds if extent is None else extent
        z = domain(E, res)
        with np.errstate(all="ignore"):
            w = np.asarray(f(z), dtype=complex)
        mod = np.abs(w)
        height = np.log1p(mod) if log else mod
        cap = self.view.bounds if zmax is None else zmax
        height = np.clip(np.nan_to_num(height, nan=cap, posinf=cap), 0, cap)
        colors = (
            colorize(w, **kw)
            if kw
            else colorize(
                w,
                phase_contours=(scheme in ("phase", "enhanced")),
                modulus_contours=(scheme in ("modulus", "enhanced")),
            )
        )
        return self.surface(z.real, z.imag, height, colors=colors)

    # ── Riemann surfaces ─────────────────────────────────────
    def riemann_root(
        self,
        n=2,
        *,
        radius=None,
        nr=60,
        ntheta=241,
        scheme="enhanced",
        height="im",
        height_scale=1.0,
    ):
        """The Riemann surface of the multivalued ``z^{1/n}`` (``√z`` for n=2).

        Parametrized by the *value* ``w`` over a disk (so ``z = wⁿ`` covers the base plane n-to-1 and
        the sheets join smoothly at the branch point). Height is ``Im w`` (or ``"re"``); colored by the
        phase of ``w``. The classic self-intersecting "parking ramp" for n=2.
        """
        R = self.view.bounds if radius is None else radius
        rho = np.linspace(0, R ** (1.0 / n), nr)
        phi = np.linspace(0, 2 * np.pi, ntheta)  # one turn of w = n sheets of z
        RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
        w = RHO * np.exp(1j * PHI)
        z = w**n
        h = (w.imag if height == "im" else w.real) * height_scale
        return self.surface(z.real, z.imag, h, colors=_scheme_colors(w, scheme))

    def riemann_log(
        self,
        *,
        radius=None,
        sheets=3,
        rmin=0.12,
        nr=60,
        ntheta=241,
        scheme="enhanced",
        height_scale=0.5,
    ):
        """The Riemann surface of ``log z`` — the infinite spiral **helicoid** (``sheets`` turns shown).

        Parametrized by ``w = log z`` over a rectangle; ``z = e^w``, height is the winding ``Im w``
        (the sheet), colored by the phase of ``z`` so each 2π turn cycles the hue.
        """
        R = self.view.bounds if radius is None else radius
        u = np.linspace(np.log(rmin), np.log(R), nr)
        v = np.linspace(0, 2 * np.pi * sheets, ntheta)
        U, V = np.meshgrid(u, v, indexing="ij")
        z = np.exp(U + 1j * V)
        return self.surface(
            z.real, z.imag, V * height_scale, colors=_scheme_colors(z, scheme)
        )

    # ── output ───────────────────────────────────────────────
    def display(self):
        return get_backend(self._backend).render(
            P.Scene(self.view, list(self.primitives))
        )

    def save(self, path):
        return get_backend(self._backend).render(
            P.Scene(self.view, list(self.primitives)), save=path
        )
