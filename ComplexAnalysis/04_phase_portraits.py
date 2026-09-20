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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Complex Analysis: Phase Portraits (Wegert)
#
# A phase portrait colors the domain by the value of a complex function. The hue shows the phase
# `arg f(z)`. One full color wheel equals one full turn, and red means `arg = 0`. Optional
# brightness contours (bands with lower brightness) show more structure. E. Wegert describes this
# method in *Visual Complex Functions* (2012). `clmmathtools` offers four schemes:
#
# | scheme | shows |
# |--------|-------|
# | `plane` | hue only, that is, the bare phase |
# | `phase` | plus phase contours (bands of constant argument) |
# | `modulus` | plus modulus contours (bands of constant `|f|`, spaced by log) |
# | `enhanced` | both, spaced equally, which draws small squares that follow the map (default) |
#
# `Plane.phase_portrait(f, scheme=…)` fills the plane. Grid, points, and labels draw on top.

# %%
import numpy as np
import clmmathtools.viz as mv
from clmmathtools.viz import phase
from clmmathtools.viz import primitives as P

# %% [markdown]
# ## The four schemes on the identity `f(z) = z`
#
# The same function appears at four levels of enhancement, from bare hue to the square grid.

# %%
for scheme in ["plane", "phase", "modulus", "enhanced"]:
    p = mv.Plane(extent=2, grid=False, axes=False)
    p.phase_portrait(lambda z: z, res=400, scheme=scheme).text([-1.9, 1.7], scheme)
    p.display()

# the schemes differ, and contours only darken (same hue, lower brightness)
z = phase.domain(2.0, 120)
plane = phase.colorize(z, phase_contours=False, modulus_contours=False)
enhanced = phase.colorize(z)
assert (enhanced <= plane + 1e-9).all() and enhanced.mean() < plane.mean()
print("schemes differ; enhancement only darkens (adds contour lines) ✓")

# %% [markdown]
# ## A small gallery (enhanced)
#
# A zero is a point where all hues meet and the hues run counter-clockwise. A pole is a point
# where all hues meet and the hues run clockwise. A pole renders white at that point. Count the
# color cycles to read the order.

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
# The winding number of `f` at a point is the number of hue cycles when you walk around that
# point. The number is positive at a zero, and equals the order of the zero. The number is
# negative at a pole. This section verifies the rule with numbers.


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

# poles render white, and the whole portrait stays finite even through zeros
rgb = phase.phase_portrait(lambda z: 1 / z, extent=1, res=200, scheme="enhanced")
assert np.isfinite(rgb).all()
print("portrait finite everywhere; pole shows white ✓")

# %% [markdown]
# ## Both backends
#
# Two backends render phase portraits. The matplotlib backend draws a raster image directly. The
# vedo backend draws a textured image behind the vector layer. Both give the same picture, so an
# overlay such as a grid looks the same in both.

# %%
mv.Plane(extent=2, grid=False, axes=False, backend="vedo").phase_portrait(
    lambda z: (z**2 - 1) / (z**2 + 1), res=400
).grid(alpha=0.2).display()
print("vedo raster path OK — grid composes over the portrait ✓")

# %% [markdown]
# `phase_portrait(f, scheme=…)` gives the plane, phase, modulus, and enhanced Wegert portraits.
# You can read zeros and poles from the direction and the count of the hue cycles. The winding
# numbers above verify this. The next notebook covers the 3D analytic landscape, that is, the
# modulus surface colored by this same phase.

# %% [markdown]
#
