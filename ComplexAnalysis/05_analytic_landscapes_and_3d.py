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
# # clmmathtools examples 5: the 3D scene, analytic landscapes and 3D vector fields
#
# `Space3D` is the 3D version of `Plane`, with a perspective view. It keeps the same design: the
# code records backend-neutral primitives, and the backend renders them. It supports `Arrow3D`,
# `Line3D`, `Points3D`, and `Surface`. The backend is vedo (default) or matplotlib
# (`backend="mpl"`). Two main builders exist:
#
# - `Space3D.landscape(f)` draws the analytic landscape. It is the surface `|f(z)|` colored by
#   `arg f(z)`, the same coloring as a phase portrait. The flat colors of a portrait become 3D
#   terrain.
# - `Space3D.field(f)` draws a 3D vector field `f: ℝ³→ℝ³` as a lattice of arrows. All arrows have
#   the same length, and color shows magnitude. This is the same convention as the 2D
#   `Plane.field`.

# %%
import numpy as np
import clmmathtools.viz as mv
from clmmathtools.viz import primitives as P

# %% [markdown]
# ## Analytic landscapes
#
# A zero looks like a valley that touches the floor. A pole looks like a tower, clipped at `zmax`.
# The hue winds around each point, exactly as in the flat phase portrait. The landscape only lifts
# `|f|` into the third dimension.

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
# `Space3D.field(f)` samples a 3D lattice. All arrows have the same length, and color shows the
# magnitude. `f` can be a 3×3 matrix (the linear field `x ↦ Mx`) or a callable `(N,3)->(N,3)`.


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
# A multi-valued function becomes single-valued on its Riemann surface. The code parametrizes the
# surface by the value `w`, so the sheets join smoothly at the branch point. Color shows the phase.
#
# - `riemann_root(n)` draws the surface of `z^{1/n}`. There `z = wⁿ` covers the base plane n times.
#   For `n=2` (`√z`) you get the classic two-sheet "parking ramp" that crosses itself.
# - `riemann_log()` draws the surface of `log z`. It is an infinite helicoid (a spiral staircase).
#   Each 2π turn is a new sheet, and the hue cycles once per turn.

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
# The 3D scene renders under both backends. Vedo draws VTK meshes and arrows. Matplotlib draws
# with mplot3d.

# %%
mv.Space3D(bounds=2.5, backend="mpl").landscape(
    lambda z: (z**2 - 1) / (z**2 + 1), res=300, zmax=3
).display()
print("matplotlib mplot3d path OK — same landscape ✓")

# %% [markdown]
# `Space3D` lifts the library into 3D. `landscape(f)` gives the analytic landscape form of a phase
# portrait: the surface `|f|` colored by `arg f`. `field(f)` draws 3D vector fields with the same
# convention as 2D: all arrows have the same length, and color shows magnitude. Both work on both
# backends. `riemann_root` and `riemann_log` draw multi-sheeted Riemann surfaces (√z, ∛z, log z)
# colored by phase.

# %% [markdown]
#

# %% [markdown]
#
