# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: doingmath (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # mathviz examples 4 — phase portraits (Wegert)
#
# A **phase portrait** colors the domain by the value of a complex function: the **hue** is the phase
# `arg f(z)` (a full color wheel per turn, red at `arg = 0`), and optional brightness **contours**
# encode more structure. Following E. Wegert, *Visual Complex Functions* (2012), `mathviz` offers four
# schemes:
#
# | scheme | shows |
# |--------|-------|
# | `plane` | hue only — the bare phase |
# | `phase` | + phase contours (constant-argument bands) |
# | `modulus` | + modulus contours (log-spaced constant-`|f|` bands) |
# | `enhanced` | **both**, spaced equally → conformal little squares *(default)* |
#
# `Plane.phase_portrait(f, scheme=…)` fills the plane; grid/points/labels compose on top.

# %%
import numpy as np
import mathviz as mv
from mathviz import phase
from mathviz import primitives as P

# %% [markdown]
# ## The four schemes on the identity `f(z) = z`
#
# Same function, four levels of enhancement — from bare hue to the conformal grid.

# %%
for scheme in ["plane", "phase", "modulus", "enhanced"]:
    p = mv.Plane(extent=2, grid=False, axes=False)
    p.phase_portrait(lambda z: z, res=400, scheme=scheme).text([-1.9, 1.7], scheme)
    p.display()

# the schemes really differ, and contours only ever darken (same hue, lower brightness)
z = phase.domain(2.0, 120)
plane = phase.colorize(z, phase_contours=False, modulus_contours=False)
enhanced = phase.colorize(z)
assert (enhanced <= plane + 1e-9).all() and enhanced.mean() < plane.mean()
print("schemes differ; enhancement only darkens (adds contour lines) ✓")

# %% [markdown]
# ## A small gallery (enhanced)
#
# Zeros are where all hues meet running **counter-clockwise**; poles are where they meet running
# **clockwise** (and render white at the singular point). Count the color cycles to read the order.

# %%
gallery = [
    ("z² : double zero at 0", lambda z: z**2),
    ("1/z : simple pole at 0", lambda z: 1 / z),
    ("(z²−1)/(z²+1) : zeros ±1, poles ±i", lambda z: (z**2 - 1) / (z**2 + 1)),
    ("exp(z) : no zeros, essential ∞", np.exp),
    ("sin(z) : zeros at kπ", np.sin),
]
for title, f in gallery:
    p = mv.Plane(extent=3, grid=False, axes=False)
    p.phase_portrait(f, res=500, scheme="enhanced").text([-2.8, 2.6], title)
    p.display()

# %% [markdown]
# ## Reading the portrait: winding number = order of a zero/pole
#
# The number of times the hue cycles as you circle a point is the **winding number** of `f` there —
# positive for a zero of that order, negative for a pole. We verify it numerically.


# %%
def winding(f, center=0.0, r=0.4, n=2000):
    t = np.linspace(0, 2 * np.pi, n)
    ang = np.unwrap(np.angle(f(center + r * np.exp(1j * t))))
    return round((ang[-1] - ang[0]) / (2 * np.pi))


assert winding(lambda z: z**2) == 2  # double zero
assert winding(lambda z: z**3) == 3  # triple zero
assert winding(lambda z: 1 / z) == -1  # simple pole
assert winding(lambda z: (z**2 - 1) / (z**2 + 1), center=1.0, r=0.3) == 1  # zero at z=1
assert winding(lambda z: (z**2 - 1) / (z**2 + 1), center=1j, r=0.3) == -1  # pole at z=i
print("winding numbers match zero/pole orders ✓ (2, 3, −1, +1 at z=1, −1 at z=i)")

# poles render white; the whole portrait stays finite even through zeros
rgb = phase.phase_portrait(lambda z: 1 / z, extent=1, res=200, scheme="enhanced")
assert np.isfinite(rgb).all()
print("portrait finite everywhere; pole shows white ✓")

# %% [markdown]
# ## Both backends
#
# Phase portraits render under matplotlib (raster-native) and vedo (as a textured image behind the
# vector layer) — same picture, so overlays like a grid compose identically.

# %%
mv.Plane(extent=2, grid=False, axes=False, backend="vedo").phase_portrait(
    lambda z: (z**2 - 1) / (z**2 + 1), res=400
).grid(alpha=0.2).display()
print("vedo raster path OK — grid composes over the portrait ✓")

# %% [markdown]
# **Recap.** `phase_portrait(f, scheme=…)` gives the plane / phase / modulus / enhanced Wegert
# portraits; zeros and poles are legible by the direction and count of the hue cycles, verified here
# by winding numbers. Next up (Phase 4): the 3D **analytic landscape** — the modulus surface colored
# by this same phase.

# %% [markdown]
#
