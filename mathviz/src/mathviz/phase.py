"""Wegert phase portraits — domain coloring of complex functions.

A phase portrait colors each point ``z`` of the domain by the value ``f(z)``: the **hue** encodes
the phase ``arg f(z)`` (a full color wheel per turn, red at ``arg = 0``), and optional brightness
**contours** encode structure, following E. Wegert, *Visual Complex Functions* (2012):

* ``"plain"``    — hue only (the bare phase portrait).
* ``"phase"``    — hue + **phase contours** (isochromatic bands; lines of constant argument).
* ``"modulus"``  — hue + **modulus contours** (log-spaced; lines of constant ``|f|``).
* ``"enhanced"`` — **both**, spaced equally so the cells are conformal little squares. *(default)*

Zeros become the points where all hues meet; poles/branch points (non-finite ``f``) render white.
Everything is a pure NumPy → RGB computation, so it drops straight onto ``Plane.raster``.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

_SCHEMES = {
    "plain": dict(phase_contours=False, modulus_contours=False),
    "phase": dict(phase_contours=True, modulus_contours=False),
    "modulus": dict(phase_contours=False, modulus_contours=True),
    "enhanced": dict(phase_contours=True, modulus_contours=True),
}


def domain(extent, res=600):
    """The complex sample grid over the square [-extent, extent]² (row 0 = bottom, y increasing)."""
    xs = np.linspace(-extent, extent, res)
    re, im = np.meshgrid(xs, xs)
    return re + 1j * im


def _frac_integral(x):
    """``∫₀ˣ frac(u) du = ⌊x⌋/2 + frac(x)²/2`` — each whole period contributes exactly ``1/2``."""
    floor = np.floor(x)
    return floor * 0.5 + (x - floor) ** 2 * 0.5


def _sawtooth(t, low=0.65, width=None):
    """Brightness ramp ``low`` → 1 across each unit band; the reset edge reads as a contour line.

    ``width`` (optional) is how much ``t`` changes across one screen pixel. Given it, we return the
    **exact average of the ramp over that pixel** instead of a point sample — analytic anti-aliasing.
    The ramp's reset is a genuine discontinuity, so point-sampling it aliases at *any* resolution;
    box-filtering it does not. As ``width → 0`` this reduces to ``frac(t)``; as ``width → ∞`` it tends
    to ``1/2``, so contours packed finer than a pixel fade to flat tone rather than moiré.
    """
    t = np.asarray(t, dtype=float)
    if width is None:
        return low + (1.0 - low) * (t % 1.0)

    w = np.abs(np.asarray(width, dtype=float))
    w = np.where(np.isfinite(w), w, 1e6)  # a pole's neighbourhood → fully averaged
    w = np.clip(w, 0.0, 1e6)
    tiny = w < 1e-9  # avoid 0/0; the limit is the plain point sample
    safe = np.where(tiny, 1.0, w)
    avg = (_frac_integral(t + safe / 2) - _frac_integral(t - safe / 2)) / safe
    avg = np.where(tiny, t % 1.0, avg)
    return low + (1.0 - low) * np.clip(np.nan_to_num(avg, nan=0.5), 0.0, 1.0)


def colorize(
    w,
    *,
    phase_contours=True,
    modulus_contours=True,
    steps=6,
    sat=1.0,
    low=0.65,
    d_logf=None,
):
    """Color an array of complex values ``w`` → an (H, W, 3) RGB array in [0, 1].

    ``steps`` sets the number of contour bands per turn of phase; the modulus contours use the same
    spacing in ``log|f|`` so the enhanced cells are (conformally) square.

    ``d_logf`` — how much ``|log f|`` changes across one screen pixel, i.e. ``|f'/f| · pixel_size`` —
    switches on analytic anti-aliasing (see :func:`_sawtooth`). **Both** contour families have that
    same gradient magnitude, since ``d(log f) = (f'/f) dz`` splits into ``d log|f|`` (real part) and
    ``d arg f`` (imaginary part) — which is exactly why the ``enhanced`` cells come out square.
    """
    import matplotlib.colors as mcolors

    w = np.asarray(w, dtype=complex)
    finite = np.isfinite(w)
    wf = np.where(finite, w, 0.0)

    arg = np.angle(wf)
    hue = (arg % TWO_PI) / TWO_PI  # [0,1): red at arg=0, cycling CCW
    val = np.ones(w.shape, dtype=float)
    band = None if d_logf is None else np.asarray(d_logf, float) * steps / TWO_PI

    if phase_contours:
        val *= _sawtooth(hue * steps, low, band)
    if modulus_contours:
        with np.errstate(divide="ignore"):
            log_mod = np.log(np.abs(wf))
        val *= _sawtooth(log_mod * steps / TWO_PI, low, band)

    val = np.nan_to_num(np.clip(val, 0.0, 1.0), nan=1.0)
    sat_arr = np.full(w.shape, sat, dtype=float)
    rgb = mcolors.hsv_to_rgb(np.stack([hue, sat_arr, val], axis=-1))
    rgb[~finite] = 1.0  # poles / branch points → white
    return rgb


def _d_logf(w):
    """``|f'/f| · pixel_size`` — the per-pixel change in ``log f``, from the samples we already have.

    For holomorphic ``f`` the complex derivative *is* the partial along ``x`` (axis 1 of the sample
    grid), so one finite difference of ``w`` recovers ``f'`` with **no extra evaluation of ``f``**.
    ``np.gradient`` at unit spacing already returns the change *per pixel*, which is the quantity the
    box filter wants — no rescaling by the domain step is needed.
    """
    with np.errstate(all="ignore"):
        return np.abs(np.gradient(w, axis=1) / w)


def phase_portrait(f, extent=3.0, res=600, scheme="enhanced", aa=True, **kw):
    """Compute the RGB phase portrait of ``f`` over [-extent, extent]². Returns an (res, res, 3) array.

    ``scheme`` is one of 'plain', 'phase', 'modulus', 'enhanced'; extra keywords (``steps``, ``sat``,
    ``low``) pass through to :func:`colorize`.

    ``aa=True`` (default) **anti-aliases the contour bands analytically**: the brightness ramp resets
    discontinuously at each band edge, and point-sampling a discontinuity aliases into a staircase at
    every resolution. We box-filter it exactly over each pixel instead, which costs one
    ``np.gradient`` and no extra evaluations of ``f``. Raising ``res`` alone does *not* fix this —
    though ``res`` should still be at least the width in pixels that the raster is displayed at.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose one of {sorted(_SCHEMES)}")
    z = domain(extent, res)
    with np.errstate(all="ignore"):
        w = np.asarray(f(z), dtype=complex)
    return colorize(w, **{**_SCHEMES[scheme], **kw, "d_logf": _d_logf(w) if aa else None})
