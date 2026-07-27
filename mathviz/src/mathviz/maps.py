"""Push-forward: send the plane through a map and draw the image.

This is the one operation that unifies three things that look separate across the notebooks:

* ``apply_matrix(M)`` — a **linear** map (linear algebra): the coordinate grid warps to a lattice.
* ``show_operator(f)`` — a geometric **operator** (project/rotate/shear): before/after overlay.
* ``apply_complex(g)`` — a **complex** function (conformal map): the grid warps to curved lines.

All three are ``push(plane, f)``: normalize the map to a point-map ``(N,2) -> (N,2)``, sample the
domain grid (and optional probe shape / basis) finely, transform the samples, and draw original
(faint) beside image (bold). Non-finite images (poles, branch points) split the drawn curves so a
singularity cleanly breaks the line instead of streaking to infinity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import palette

# ── probe shapes (reused for "watch what the map does to a recognizable figure") ──
#: What every ``probe_shape`` argument accepts: ``None`` for no probe, or anything ``np.asarray``
#: turns into an ``(N,2)`` array of ``(x, y)`` points — an ndarray, a list of pairs, a tuple of
#: tuples. Repeat the first point at the end to close the outline.
ProbeShape = np.ndarray | Sequence[Sequence[float]] | None

#: An asymmetric "flag on a pole" — chirality/orientation are instantly legible.
FLAG = np.array(
    [[0.0, 0.0], [0.0, 2.0], [1.2, 1.6], [0.6, 1.2], [0.0, 1.2], [0.0, 0.0]]
)
#: The unit square [0,1]² (its image area = det, for linear maps).
UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])


def unit_circle(n: int = 240) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)]


# ── the fundamental linear operations of the plane ────────────────────────────────
# GL(2, ℝ) has two connected components, separated by the det = 0 wall:
#   det > 0 — rotation, scaling, shear: the identity's component, smoothly reachable from I.
#   det < 0 — reflection: the *other* component; any path from I must cross det = 0.
#   det = 0 — projection: on the wall itself, not invertible.
# See mathviz.animate.matrix_path for what that means for animating each one.


def rotation(theta: float) -> np.ndarray:
    """Rotation by ``theta`` radians about the origin. ``det = +1``, no real eigenvalues (θ ∉ πℤ)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling(sx: float, sy: float | None = None) -> np.ndarray:
    """``diag(sx, sy)``; ``sy`` defaults to ``sx`` (a uniform dilation). ``det = sx·sy``.

    ``scaling(k, 1/k)`` is the **squeeze** (hyperbolic rotation): area-preserving but not a rotation.
    """
    return np.diag([float(sx), float(sx if sy is None else sy)])


def shear(k: float, axis: str = "x") -> np.ndarray:
    """Shear of strength ``k`` parallel to ``axis``. ``det = 1``; the ``axis`` line is fixed pointwise.

    ``axis="x"`` slides points horizontally in proportion to ``y`` (``[[1, k], [0, 1]]``).
    Its only eigenvalue is 1, with a *single* eigendirection — the shear is **not diagonalizable**.
    """
    if axis == "x":
        return np.array([[1.0, float(k)], [0.0, 1.0]])
    if axis == "y":
        return np.array([[1.0, 0.0], [float(k), 1.0]])
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def reflection(theta: float = 0.0) -> np.ndarray:
    """Reflection **in** the line through the origin at angle ``theta``. ``det = -1``.

    Orientation-reversing, so it sits in the component of GL(2, ℝ) *not* containing the identity —
    no continuous path of invertible maps joins it to ``I``. Eigenvalues ``+1`` (along the mirror)
    and ``-1`` (across it).
    """
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return np.array([[c, s], [s, -c]])


def projection(theta: float = 0.0) -> np.ndarray:
    """Orthogonal projection **onto** the line through the origin at angle ``theta``. ``det = 0``.

    Singular (rank 1) and **idempotent**: ``P² = P``. Eigenvalues ``1`` (along the line) and ``0``
    (the kernel, perpendicular to it). Not invertible, so it lies on the ``det = 0`` boundary.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, c * s], [c * s, s * s]])


# ── normalizing any map to a point-map (N,2) -> (N,2) ─────────────────────────────
def from_matrix(M):
    """A 2×2 matrix as a point-map: p ↦ M p (row-vector form ``pts @ M.T``)."""
    M = np.asarray(M, dtype=float).reshape(2, 2)
    return lambda pts: np.asarray(pts, float).reshape(-1, 2) @ M.T


def from_complex(g):
    """A complex function g: ℂ→ℂ as a point-map. Non-finite outputs become NaN (they break curves)."""

    def f(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            w = np.asarray(g(pts[:, 0] + 1j * pts[:, 1]), dtype=complex)
        out = np.c_[w.real, w.imag]
        out[~np.isfinite(out).all(axis=1)] = np.nan
        return out

    return f


def as_pointmap(f):
    """Accept a 2×2 matrix, or a callable already of the form (N,2)->(N,2)."""
    if isinstance(f, np.ndarray) or (np.ndim(f) == 2):
        return from_matrix(f)
    return f


# ── the domain grid, sampled finely enough for curved images ──────────────────────
def domain_gridlines(extent, step=1.0, samples=100):
    E = float(extent)
    lines = []
    ticks = np.arange(-np.floor(E / step) * step, E + 1e-9, step)
    s = np.linspace(-E, E, samples)
    for t in ticks:
        lines.append(np.c_[s, np.full_like(s, t)])  # horizontal line y = t
        lines.append(np.c_[np.full_like(s, t), s])  # vertical line   x = t
    return lines


def _finite_runs(pts):
    """Split a polyline at non-finite points into its maximal finite runs (length ≥ 2)."""
    pts = np.asarray(pts, float)
    ok = np.isfinite(pts).all(axis=1)
    runs, i, n = [], 0, len(pts)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= 2:
            runs.append(pts[i:j])
        i = j
    return runs


# ── the core: push the plane through a map and draw ───────────────────────────────
def push(
    plane,
    f,
    *,
    color=palette.ORANGE,
    width=2.0,
    grid=True,
    step=1.0,
    samples=100,
    faint=True,
    faint_color=palette.FAINT,
    probe_shape: ProbeShape = None,
    probe_shape_color: int | None = None,
    basis=False,
    basis_labels=("f(e1)", "f(e2)"),
):
    """Draw the image of the domain (grid / probe shape / basis) under ``f`` onto ``plane``.

    ``f`` may be a 2×2 matrix or a point-map ``(N,2)->(N,2)``. Returns ``plane`` (chainable).

    ``probe_shape`` is the recognizable figure carried through the map — ``None`` for no probe, or
    anything ``np.asarray`` turns into an ``(N,2)`` array of points (a list of ``(x, y)`` pairs, an
    ``(N,2)`` ndarray, …), such as :data:`FLAG` or :data:`UNIT_SQUARE`. Repeat the first point at the
    end to close it. ``probe_shape_color`` colors the probe's *image* — it defaults to ``color``,
    the grid image's color; the faint pre-image follows ``faint_color`` either way.
    """
    fmap = as_pointmap(f)
    E = plane.view.extent

    if grid:
        for ln in domain_gridlines(E, step=step, samples=samples):
            if faint:
                plane.curve(ln, color=faint_color, width=1.0, alpha=0.9)
            for run in _finite_runs(fmap(ln)):
                plane.curve(run, color=color, width=width)

    if probe_shape is not None:
        probe_shape = np.asarray(probe_shape, float).reshape(-1, 2)
        if faint:
            plane.curve(probe_shape, color=faint_color, width=1.6, alpha=0.9)
        pc = color if probe_shape_color is None else probe_shape_color
        for run in _finite_runs(fmap(probe_shape)):
            plane.curve(run, color=pc, width=width + 1.0)

    if basis:
        for e, bc, lab in (
            ([1.0, 0.0], palette.RED, basis_labels[0]),
            ([0.0, 1.0], palette.GREEN, basis_labels[1]),
        ):
            img = fmap(np.array([e]))[0]
            if np.isfinite(img).all():
                plane.vector(img, color=bc, label=lab)
    return plane


def apply_matrix(plane, M, *, basis=True, basis_labels=("M e1", "M e2"), **kw):
    """Linear-algebra view: warp the coordinate grid by the 2×2 matrix ``M`` and draw its columns."""
    return push(plane, from_matrix(M), basis=basis, basis_labels=basis_labels, **kw)


def apply_complex(plane, g, **kw):
    """Complex-analysis view: the conformal image of the grid under g: ℂ→ℂ (the warped mesh)."""
    return push(plane, from_complex(g), **kw)


def show_operator(plane, f, *, probe_shape: ProbeShape = None, **kw):
    """Geometric-operator view: faint original grid (+probe) beside their bold image under ``f``.

    ``probe_shape`` is off by default; pass :data:`FLAG` (asymmetric, so a flip or shear is legible
    at a glance), :data:`UNIT_SQUARE`, or any array of points — see :func:`push`.
    """
    return push(plane, f, probe_shape=probe_shape, **kw)


def vector_field(
    plane,
    f,
    *,
    n=21,
    color=palette.BLUE,
    width=2.0,
    scale=None,
    normalize=True,
    cmap="viridis",
):
    """Draw the vector field ``f`` as a grid of arrows.

    ``f`` maps sample points to vectors: a 2×2 matrix (the linear field x ↦ Mx) or a point-map
    ``(N,2)->(N,2)``.

    **By default the arrows are drawn at a uniform length (direction only) and the magnitude is
    encoded by color** (``normalize=True``, ``cmap="viridis"``) — so a dense field stays legible and
    the magnitude is read from the colormap rather than from overlapping arrow lengths. Set
    ``normalize=False`` to make arrow length proportional to magnitude (auto-scaled so the longest
    spans ~one grid cell, or use ``scale``); set ``cmap=None`` to draw every arrow in the single
    ``color``. Returns ``plane``.
    """
    fmap = as_pointmap(f)
    E = plane.view.extent
    xs = np.linspace(-E, E, n)
    pts = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    vec = fmap(pts)
    mag = np.hypot(vec[:, 0], vec[:, 1])
    ok = np.isfinite(mag) & (mag > 0)
    cell = (2 * E) / (n - 1)

    if normalize:
        disp = np.zeros_like(vec)
        disp[ok] = vec[ok] / mag[ok, None] * (cell * 0.45)
    else:
        peak = np.nanmax(mag[ok]) if ok.any() else 1.0
        s = scale if scale is not None else (cell * 0.9) / peak
        disp = vec * s

    cmap_fn = norm = None
    if cmap is not None:
        import matplotlib as mpl
        import matplotlib.cm as cm

        cmap_fn = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else mpl.colormaps[cmap]
        lo, hi = float(np.nanmin(mag[ok])), float(np.nanmax(mag[ok]))
        norm = mpl.colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)

    for i in range(len(pts)):
        if not ok[i]:
            continue
        c = color
        if cmap_fn is not None:
            r, g, b, _ = cmap_fn(norm(mag[i]))
            c = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        plane.vector(disp[i], origin=pts[i], color=c, width=width)
    return plane
