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


def _sawtooth(t, low=0.65):
    """Brightness ramp ``low`` → 1 across each unit band; the reset edge reads as a contour line."""
    return low + (1.0 - low) * (np.asarray(t, float) % 1.0)


def colorize(
    w, *, phase_contours=True, modulus_contours=True, steps=6, sat=1.0, low=0.65
):
    """Color an array of complex values ``w`` → an (H, W, 3) RGB array in [0, 1].

    ``steps`` sets the number of contour bands per turn of phase; the modulus contours use the same
    spacing in ``log|f|`` so the enhanced cells are (conformally) square.
    """
    import matplotlib.colors as mcolors

    w = np.asarray(w, dtype=complex)
    finite = np.isfinite(w)
    wf = np.where(finite, w, 0.0)

    arg = np.angle(wf)
    hue = (arg % TWO_PI) / TWO_PI  # [0,1): red at arg=0, cycling CCW
    val = np.ones(w.shape, dtype=float)

    if phase_contours:
        val *= _sawtooth(hue * steps, low)
    if modulus_contours:
        with np.errstate(divide="ignore"):
            log_mod = np.log(np.abs(wf))
        val *= _sawtooth(log_mod * steps / TWO_PI, low)

    val = np.nan_to_num(np.clip(val, 0.0, 1.0), nan=1.0)
    sat_arr = np.full(w.shape, sat, dtype=float)
    rgb = mcolors.hsv_to_rgb(np.stack([hue, sat_arr, val], axis=-1))
    rgb[~finite] = 1.0  # poles / branch points → white
    return rgb


def phase_portrait(f, extent=3.0, res=600, scheme="enhanced", **kw):
    """Compute the RGB phase portrait of ``f`` over [-extent, extent]². Returns an (res, res, 3) array.

    ``scheme`` is one of 'plain', 'phase', 'modulus', 'enhanced'; extra keywords (``steps``, ``sat``,
    ``low``) pass through to :func:`colorize`.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose one of {sorted(_SCHEMES)}")
    z = domain(extent, res)
    with np.errstate(all="ignore"):
        w = np.asarray(f(z), dtype=complex)
    return colorize(w, **{**_SCHEMES[scheme], **kw})
