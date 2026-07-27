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
# # mathviz examples 5 — the 3D scene: analytic landscapes & 3D vector fields
#
# `Space3D` is the perspective 3D sibling of `Plane`: the same "record backend-neutral primitives, let a
# backend render them" design, now with `Arrow3D` / `Line3D` / `Points3D` / `Surface`, on **vedo**
# (default) or **matplotlib** (`backend="mpl"`). Two headline builders:
#
# - `Space3D.landscape(f)` — the **analytic landscape**: the surface `|f(z)|` colored by `arg f(z)`
#   (the same coloring as a phase portrait), so a portrait's flat colors become 3D terrain.
# - `Space3D.field(f)` — a **3D vector field** `f: ℝ³→ℝ³` as a lattice of arrows (uniform length,
#   magnitude by color — the same convention as the 2D `Plane.field`).

# %%
import numpy as np
import mathviz as mv
from mathviz import primitives as P

# %% [markdown]
# ## Analytic landscapes
#
# Zeros are valleys touching the floor; poles are towers (clipped at `zmax`). The hue winds around
# each, exactly as in the flat phase portrait — the landscape just lifts `|f|` into the third dimension.

# %%
mv.Space3D(bounds=2.5).landscape(
    lambda z: z**2, res=2 * 110, zmax=3
).display()  # double zero at 0
mv.Space3D(bounds=2.5).landscape(
    lambda z: (z**2 - 1) / (z**2 + 1), res=120, zmax=3
).display()  # zeros ±1, poles ±i
mv.Space3D(bounds=3).landscape(
    np.sin, res=2 * 130, zmax=4, log=True
).display()  # zeros at kπ (log height)

# a landscape is a colored Surface capped at zmax and finite even through poles
s = mv.Space3D(bounds=1).landscape(lambda z: 1 / z, res=60, zmax=3)
surf = s.primitives[-1]
assert isinstance(surf, P.Surface) and surf.colors.shape == (60, 60, 3)
assert np.isfinite(surf.Z).all() and surf.Z.max() <= 3 + 1e-9
print("landscape is a colored surface, finite and capped at zmax ✓")

# %% [markdown]
# ## 3D vector fields
#
# `Space3D.field(f)` samples a 3D lattice; arrows are uniform length with magnitude shown by color.
# `f` is a 3×3 matrix (linear field `x ↦ Mx`) or a callable `(N,3)->(N,3)`.


# %%
# a rotation about z plus a gentle lift along z:  f(x,y,z) = (−y, x, 0.3 z)
def swirl(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.column_stack([-y, x, 0.3 * z])


mv.Space3D(bounds=3).field(swirl, n=6).display()

# a pure source (linear field x ↦ x): arrows point radially outward, brighter with radius
mv.Space3D(bounds=3).field(np.eye(3), n=5).display()

arrows = [
    a
    for a in mv.Space3D(bounds=2).field(swirl, n=4).primitives
    if isinstance(a, P.Arrow3D)
]
lens = [np.linalg.norm(a.head - a.tail) for a in arrows]
assert np.allclose(lens, lens[0])  # uniform length by default
print(f"{len(arrows)} arrows, uniform length {lens[0]:.3f}, magnitude by color ✓")

# %% [markdown]
# ## Riemann surfaces
#
# A multi-valued function becomes single-valued on its **Riemann surface**. We parametrize by the
# *value* `w` (so the sheets join smoothly at the branch point) and color by phase.
#
# - `riemann_root(n)` — the surface of `z^{1/n}`: `z = wⁿ` covers the base plane n-to-1. For `n=2`
#   (`√z`) it is the classic self-intersecting two-sheet "parking ramp".
# - `riemann_log()` — the surface of `log z`: an infinite **helicoid** (spiral staircase), each 2π
#   turn a new sheet, the hue cycling once per turn.

# %%
mv.Space3D(bounds=2.5, elev=18, azim=-70).riemann_root(
    2, nr=70, ntheta=280
).display()  # √z
mv.Space3D(bounds=2.5, elev=18, azim=-70).riemann_root(
    3, nr=70, ntheta=300
).display()  # ∛z: 3 sheets
mv.Space3D(bounds=3, elev=20, azim=-70).riemann_log(
    sheets=3, ntheta=300
).display()  # log z helicoid

# √z has two sheets: height = Im(w) spans below and above the base plane
surf = mv.Space3D(bounds=2).riemann_root(2, nr=40, ntheta=160).primitives[-1]
assert surf.Z.min() < 0 < surf.Z.max()
# log z is a rising helicoid: height = winding angle, up 2π per sheet
lsurf = mv.Space3D(bounds=2).riemann_log(sheets=2, height_scale=1.0).primitives[-1]
assert np.isclose(lsurf.Z.min(), 0) and lsurf.Z.max() > 2 * np.pi
print("√z: two sheets (±height) ✓   log z: helicoid rising 2π per turn ✓")

# %% [markdown]
# ## Same scene, either backend
#
# The 3D scene renders under vedo (VTK meshes/arrows) and matplotlib (mplot3d) alike.

# %%
mv.Space3D(bounds=2.5, backend="mpl").landscape(
    lambda z: (z**2 - 1) / (z**2 + 1), res=300, zmax=3
).display()
print("matplotlib mplot3d path OK — same landscape ✓")

# %% [markdown]
# **Recap.** `Space3D` lifts the library into 3D: `landscape(f)` is the analytic-landscape form of a
# phase portrait (surface `|f|` colored by `arg f`), and `field(f)` draws 3D vector fields with the
# same uniform-length, color-by-magnitude convention as 2D — on both backends. And `riemann_root` /
# `riemann_log` draw multi-sheeted **Riemann surfaces** (√z, ∛z, log z) colored by phase.

# %% [markdown]
#

# %% [markdown]
#
