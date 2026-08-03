"""Parameter sweeps: a scene that depends on ``t ∈ [0, 1]`` becomes a scrubber or a GIF.

Every animation here is driven by a **single parameter** ``t`` running 0 → 1, and a *builder*
``build(t) -> Plane | Space3D`` that draws the scene at that instant. Frames are pre-rendered
offscreen to PNG bytes and then swapped by an ipywidgets ``Play``/slider, so scrubbing is instant and
nothing depends on a live 3D notebook backend.

Animating a **linear map** ``I → M`` needs a path of matrices ``M(t)`` with ``M(0) = I`` and
``M(1) = M``, and the choice of path is the whole story:

* :func:`lerp_path` — the straight line ``(1-t)I + tM`` **in matrix space**. Matrix space is not where
  the geometry lives: for ``M = -I`` (a half-turn) this line passes exactly through the *zero matrix*
  at ``t = 1/2``. The grid collapses to a point and pops back. Useful precisely to *show* that.
* :func:`polar_path` — split ``M = R P`` (SVD polar decomposition) into a rotation ``R`` and a
  symmetric positive-definite stretch ``P``, then turn the rotation angle and raise the stretch to the
  power ``t``. Always non-singular for ``det M > 0``.
* :func:`geodesic_path` — the **one-parameter subgroup** ``M(t) = exp(t·log M)``, i.e. flow along the
  Lie algebra element that generates ``M`` (the ``LieGroups/`` notebooks in this repo, applied to
  GL⁺(2, ℝ)). Also always non-singular.

Both principled paths keep ``det M(t) ≠ 0`` for every ``t``; the lerp does not. That also fixes which
of the fundamental operations (see :mod:`mathviz.maps`) each path can animate: GL(2, ℝ) has **two
connected components** split by the ``det = 0`` wall. Rotation, scaling and shear share the identity's
component and animate smoothly; a **reflection** (``det < 0``) lives in the other one and a
**projection** (``det = 0``) lives on the wall — neither is reachable, so only ``lerp`` renders them,
and the collapse you see is the mathematics rather than an artifact.

Beyond one parameter and one matrix:

* :func:`scrubber2` — two independent Play/slider axes over ``build(s, t)``, for **composing** two
  operations and watching them fail to commute.
* :func:`homotopy` — the straight-line homotopy ``g_t(z) = (1-t)z + t·g(z)`` through *holomorphic*
  maps (a convex combination of holomorphic functions is holomorphic, so every frame stays in the
  category; conformality, however, breaks wherever ``g_t'`` vanishes).
* :func:`mobius_path` — ``geodesic_path`` one field up: the one-parameter subgroup of Möbius maps
  through ``A ∈ PSL(2, ℂ)``. :func:`classify_mobius` sorts these by ``tr²`` into elliptic / parabolic /
  hyperbolic / loxodromic — the exact complex mirror of rotation / shear / squeeze / rotate-and-scale.
"""

from __future__ import annotations

import io

import numpy as np

from .maps import UNIT_SQUARE, ProbeShape, push
from .plane import Plane

__all__ = [
    "to_png",
    "render_frames",
    "render_grid",
    "scrubber",
    "scrubber2",
    "to_gif",
    "real_logm",
    "lerp_path",
    "polar_path",
    "geodesic_path",
    "matrix_path",
    "animate_matrix",
    "homotopy",
    "mobius",
    "mobius_path",
    "classify_mobius",
]


# ── frames: a scene builder → PNG bytes ───────────────────────────────────────────
def to_png(scene) -> bytes:
    """Render a ``Plane``/``Space3D`` to PNG **bytes** instead of displaying it.

    Both backends' ``save`` accept a file object, so this never touches the filesystem and never
    pushes output into the notebook — which is what makes offscreen frame rendering possible.
    """
    buf = io.BytesIO()
    scene.save(buf)
    return buf.getvalue()


def render_frames(build, n: int = 24, ts=None, ping_pong: bool = False):
    """Pre-render ``build(t)`` at ``n`` values of ``t`` spanning [0, 1]. Returns ``(ts, pngs)``.

    ``ts`` overrides the sample points (e.g. eased, or a wider range). ``ping_pong`` appends the
    reversed interior frames so a looping ``Play`` runs 0 → 1 → 0 without a jump cut.
    """
    ts = np.linspace(0.0, 1.0, n) if ts is None else np.asarray(ts, dtype=float).ravel()
    if len(ts) < 2:
        raise ValueError("need at least 2 frames")
    pngs = [to_png(build(float(t))) for t in ts]
    if ping_pong and len(ts) > 2:
        ts = np.concatenate([ts, ts[-2:0:-1]])
        pngs = pngs + pngs[-2:0:-1]
    return ts, pngs


# ── the inline scrubber ───────────────────────────────────────────────────────────
def scrubber(
    build,
    n: int = 24,
    *,
    ts=None,
    ping_pong: bool = False,
    interval: int = 80,
    label: str = "t",
    fmt: str = "{:.2f}",
):
    """A ``Play`` button + slider scrubbing ``build(t)`` over ``t ∈ [0, 1]``. Returns a widget.

    Frames are rendered **once**, up front (this is the slow step), then the slider only swaps cached
    PNG bytes — so dragging is instant regardless of how expensive the scene is. ``interval`` is the
    Play step in milliseconds. Requires ``ipywidgets``.
    """
    import ipywidgets as widgets

    ts, pngs = render_frames(build, n, ts, ping_pong)
    last = len(pngs) - 1

    img = widgets.Image(value=pngs[0], format="png")
    play = widgets.Play(min=0, max=last, step=1, interval=interval, value=0)
    slider = widgets.IntSlider(min=0, max=last, step=1, value=0, readout=False)
    readout = widgets.Label(value=f"{label} = {fmt.format(ts[0])}")
    widgets.jslink(
        (play, "value"), (slider, "value")
    )  # Play drives the slider in-browser

    def _on_frame(change):
        k = change["new"]
        img.value = pngs[k]
        readout.value = f"{label} = {fmt.format(ts[k])}"

    slider.observe(_on_frame, names="value")
    return widgets.VBox([img, widgets.HBox([play, slider, readout])])


def render_grid(build, n=(9, 9)):
    """Pre-render ``build(s, t)`` over an ``n1 × n2`` grid of [0,1]². Returns ``(ss, tt, pngs[i][j])``."""
    n1, n2 = (n, n) if isinstance(n, int) else n
    ss, tt = np.linspace(0.0, 1.0, n1), np.linspace(0.0, 1.0, n2)
    pngs = [[to_png(build(float(s), float(t))) for t in tt] for s in ss]
    return ss, tt, pngs


def scrubber2(
    build,
    n=(9, 9),
    *,
    labels=("s", "t"),
    interval: int = 120,
    fmt: str = "{:.2f}",
):
    """Two independent ``Play``+slider axes over ``build(s, t)``, for **composing** two operations.

    Renders the full ``n1 × n2`` grid of frames up front (so the cost is quadratic — keep ``n`` and
    the scene size modest), then either slider swaps a cached PNG. Requires ``ipywidgets``.
    """
    import ipywidgets as widgets

    ss, tt, pngs = render_grid(build, n)
    axes, rows = [], []
    for vals, name in zip((ss, tt), labels):
        play = widgets.Play(
            min=0, max=len(vals) - 1, step=1, interval=interval, value=0
        )
        slider = widgets.IntSlider(
            min=0, max=len(vals) - 1, step=1, value=0, readout=False
        )
        readout = widgets.Label(value=f"{name} = {fmt.format(vals[0])}")
        widgets.jslink((play, "value"), (slider, "value"))
        axes.append((slider, readout, vals, name))
        rows.append(widgets.HBox([play, slider, readout]))

    img = widgets.Image(value=pngs[0][0], format="png")

    def _refresh(_change=None):
        i, j = axes[0][0].value, axes[1][0].value
        img.value = pngs[i][j]
        for slider, readout, vals, name in axes:
            readout.value = f"{name} = {fmt.format(vals[slider.value])}"

    for slider, *_ in axes:
        slider.observe(_refresh, names="value")
    return widgets.VBox([img, *rows])


def to_gif(
    build, path, n: int = 24, *, fps: int = 20, ts=None, ping_pong: bool = False
):
    """Write the sweep to an animated GIF (portable, renders on GitHub). Returns ``path``."""
    from PIL import Image

    _, pngs = render_frames(build, n, ts, ping_pong)
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in pngs]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(round(1000 / fps)),
        loop=0,
    )
    return path


# ── matrix paths: how to get from I to M ──────────────────────────────────────────
def _singular(M, rtol: float = 1e-10) -> bool:
    """Rank-deficiency test by singular values — ``det`` alone is unreliable near zero.

    ``det(projection(π/6))`` evaluates to ``1.6e-17``, not ``0.0``, so an exact ``det == 0`` (or
    ``det <= 0``) test silently passes a projection through. Comparing ``σ_min`` to ``σ_max`` is the
    scale-invariant question actually being asked.
    """
    s = np.linalg.svd(np.asarray(M, dtype=float).reshape(2, 2), compute_uv=False)
    return bool(s[-1] <= rtol * s[0])


def _reject_unreachable(M) -> float:
    """Raise unless ``M`` can be joined to ``I`` by a path of invertible maps. Returns ``det M``."""
    det = float(np.linalg.det(np.asarray(M, dtype=float).reshape(2, 2)))
    if _singular(M):
        raise ValueError(
            f"det M = {det:.3g} ≈ 0: M is singular (e.g. a projection). Since "
            "det exp(A) = e^{tr A} > 0 strictly, *no* singular matrix has a logarithm, and it "
            'lies on the det = 0 wall no invertible path can reach. Use kind="lerp".'
        )
    if det < 0:
        raise ValueError(
            "det M < 0: M is orientation-reversing, so it lies in the other connected component "
            "of GL(2, ℝ) and no continuous path of invertible maps joins it to the identity. "
            'Use kind="lerp" to watch it degenerate through det = 0.'
        )
    return det


def real_logm(M) -> np.ndarray:
    """A **real** 2×2 matrix ``L`` with ``expm(L) == M``, for ``M`` with ``det M > 0``.

    ``scipy.linalg.logm`` returns the *principal* logarithm, whose branch cut runs along the negative
    real axis — so a matrix with negative real eigenvalues (a half-turn ``M = -I``, say) comes back
    **complex** (``logm(-I) = iπI``) even though a perfectly good real log exists (the rotation
    generator ``[[0, -π], [π, 0]]``). We take the principal log when it is real, special-case the
    negative multiples of the identity, and otherwise refuse.

    A real logarithm exists iff every negative eigenvalue has an *even* number of Jordan blocks —
    which is why ``-I`` (two 1×1 blocks for ``-1``) works but ``diag(-1, -2)`` (one block each) does
    not. Raises ``ValueError`` when no real log exists.
    """
    from scipy.linalg import expm, logm

    M = np.asarray(M, dtype=float).reshape(2, 2)
    _reject_unreachable(M)

    L = logm(M)
    if not np.allclose(L.imag, 0.0, atol=1e-10):
        # Principal branch failed — M has negative real eigenvalues.
        c = float(M[0, 0])
        if c < 0 and np.allclose(
            M, c * np.eye(2)
        ):  # M = c·I, c < 0: a half-turn, scaled
            L = np.log(-c) * np.eye(2) + np.array([[0.0, -np.pi], [np.pi, 0.0]])
        else:
            raise ValueError(
                "M has negative real eigenvalues and is not a negative multiple of I, "
                "so it has no real logarithm (odd number of Jordan blocks for a negative "
                'eigenvalue). Use kind="polar" instead.'
            )

    L = np.real(L)
    if not np.allclose(
        expm(L), M, atol=1e-8
    ):  # the repo's house style: verify, don't trust
        raise ValueError("real_logm failed: expm(L) != M")
    return L


def lerp_path(M):
    """``M(t) = (1-t)·I + t·M`` — the straight line in matrix space. **May pass through singular
    matrices** (it does, for any half-turn). Kept because seeing the collapse is the lesson."""
    M = np.asarray(M, dtype=float).reshape(2, 2)
    Id = np.eye(2)
    return lambda t: (1.0 - t) * Id + t * M


def polar_path(M):
    """``M = R·P`` (rotation × SPD stretch, via the SVD): turn ``R`` and raise ``P`` to the power ``t``.

    With ``M = U Σ Vᵀ``, take ``R = U Vᵀ`` and ``P = V Σ Vᵀ``. Then
    ``M(t) = rot(t·θ) · (V Σ^t Vᵀ)``, whose determinant ``∏ σᵢ^t`` is positive for every ``t``.
    Requires ``det M > 0``: a reflection lives in the *other* component of GL(2, ℝ), and a
    projection lives on the ``det = 0`` wall between them. Neither is reachable from ``I``.
    """
    M = np.asarray(M, dtype=float).reshape(2, 2)
    # σ_min = 0 would make σ**t jump from 1 (at t=0) to 0 (for every t>0) — a discontinuous
    # "path" that snaps to M instantly. Refuse rather than animate a lie.
    _reject_unreachable(M)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt  # the rotation factor (P = V Σ Vᵀ is the SPD stretch)
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    V = Vt.T

    def at(t):
        c, s = np.cos(t * theta), np.sin(t * theta)
        rot = np.array([[c, -s], [s, c]])
        return rot @ (V @ np.diag(S**t) @ Vt)

    return at


def geodesic_path(M):
    """``M(t) = exp(t·log M)`` — the one-parameter subgroup through ``M``.

    This is the Lie-theoretic path: ``L = log M`` is the algebra element (a constant "velocity" in
    ``gl(2, ℝ)``) whose flow for unit time is ``M``. Since ``det exp(A) = e^{tr A} > 0`` always, the
    map never degenerates. Requires a real logarithm (see :func:`real_logm`).
    """
    from scipy.linalg import expm

    L = real_logm(M)
    return lambda t: np.real(expm(t * L))


_PATHS = {"lerp": lerp_path, "polar": polar_path, "geodesic": geodesic_path}


def matrix_path(M, kind: str = "polar"):
    """Return ``t -> M(t)`` with ``M(0) = I``, ``M(1) = M``. ``kind`` ∈ {polar, geodesic, lerp}."""
    if kind not in _PATHS:
        raise ValueError(f"unknown kind {kind!r}; choose one of {sorted(_PATHS)}")
    return _PATHS[kind](M)


# ── the headline convenience ──────────────────────────────────────────────────────
def animate_matrix(
    M,
    *,
    kind: str = "polar",
    extent: float = 3.0,
    n: int = 24,
    probe_shape: ProbeShape = UNIT_SQUARE,
    probe_shape_color: int | None = None,
    size=(560, 560),
    step: float = 1.0,
    basis: bool = True,
    **scrubber_kw,
):
    """Scrub the linear map ``I → M`` along ``kind``: faint domain grid, warped image, basis arrows.

    The view is pinned to ``±extent``, so the grid warps without the camera rescaling. Straight lines
    stay straight under a linear map, so each gridline needs only its two endpoints — that is what
    keeps 24 frames fast.

    ``probe_shape`` (``None``, or any points ``np.asarray`` accepts — see :func:`mathviz.maps.push`)
    is the figure carried along; it defaults to the unit square, whose image area is ``det M(t)``.
    """
    at = matrix_path(M, kind)

    def build(t):
        plane = Plane(extent=extent, grid=False, size=size)
        return push(
            plane,
            at(t),
            probe_shape=probe_shape,
            probe_shape_color=probe_shape_color,
            basis=basis,
            basis_labels=("M(t) e1", "M(t) e2"),
            step=step,
            samples=2,  # linear ⇒ a gridline's image is determined by its endpoints
        )

    return scrubber(build, n=n, **scrubber_kw)


# ── holomorphic sweeps: paths through function space ──────────────────────────────
def homotopy(g, base=None):
    """The straight-line homotopy ``g_t(z) = (1-t)·base(z) + t·g(z)``; ``base`` defaults to ``z ↦ z``.

    A convex combination of holomorphic functions is holomorphic, so **every frame is a genuine
    holomorphic map** — the sweep stays inside the category, unlike interpolating a picture. What it
    does *not* preserve is conformality: ``g_t'`` can vanish at isolated points, and exactly there the
    right-angle crossings of the image grid break down. Returns ``t -> (z -> ℂ)``.
    """
    base = (lambda z: z) if base is None else base
    return lambda t: lambda z: (1.0 - t) * base(z) + t * g(z)


def mobius(A):
    """The Möbius transformation ``z ↦ (az + b)/(cz + d)`` of the matrix ``A = [[a,b],[c,d]]``.

    Poles (where ``cz + d = 0``) come back as ``inf``, which ``maps.from_complex`` turns into NaN so
    the drawn curves break cleanly instead of streaking.
    """
    a, b, c, d = np.asarray(A, dtype=complex).reshape(4)

    def g(z):
        z = np.asarray(z, dtype=complex)
        num, den = a * z + b, c * z + d
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den == 0, np.inf, num / den)

    return g


def _normalize_sl2(A):
    """Rescale ``A`` to ``det A = 1`` (Möbius maps only see ``A`` up to scale — ``PSL(2, ℂ)``)."""
    A = np.asarray(A, dtype=complex).reshape(2, 2)
    det = np.linalg.det(A)
    if np.isclose(det, 0):
        raise ValueError("det A = 0: not a Möbius transformation (it is constant)")
    return A / np.sqrt(det)


def classify_mobius(A) -> str:
    """``'identity' | 'elliptic' | 'parabolic' | 'hyperbolic' | 'loxodromic'``, from ``tr²`` in SL(2, ℂ).

    The exact complex analogue of the real classification of ``matrix_path``'s inputs: **elliptic** ↔
    rotation, **parabolic** ↔ shear, **hyperbolic** ↔ squeeze, **loxodromic** ↔ rotate-and-scale.
    """
    A = _normalize_sl2(A)
    if np.allclose(A, np.eye(2)) or np.allclose(A, -np.eye(2)):
        return "identity"
    tau = np.trace(A) ** 2
    if not np.isclose(tau.imag, 0.0, atol=1e-9):
        return "loxodromic"
    t = float(tau.real)
    if np.isclose(t, 4.0, atol=1e-9):
        return "parabolic"
    if 0.0 <= t < 4.0:
        return "elliptic"
    if t > 4.0:
        return "hyperbolic"
    return "loxodromic"  # tr² real and negative


def mobius_path(A):
    """``t ↦`` the Möbius map of ``exp(t·log A)`` — the one-parameter subgroup through ``A``.

    Same idea as :func:`geodesic_path`, one field up: flow along a constant velocity in
    ``sl(2, ℂ)``. ``t = 0`` is the identity map of the sphere, ``t = 1`` is ``A``. The complex
    ``logm`` needs no realness fix, so this is simpler than the real case.
    """
    from scipy.linalg import expm, logm

    L = logm(_normalize_sl2(A))
    return lambda t: mobius(expm(t * L))
