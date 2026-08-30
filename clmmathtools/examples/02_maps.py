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
# # clmmathtools examples 2 — pushing the plane through maps
#
# The unifying operation of `clmmathtools`: **push the domain grid (and shapes) through a map and draw the
# image.** The same machinery serves three worlds — linear algebra (`apply_matrix`), geometric
# operators (`show_operator`), and complex analysis (`apply_complex`) — demonstrated and asserted below.

# %%
import numpy as np
import clmmathtools.viz as mv
from clmmathtools.viz import maps
from clmmathtools.viz import primitives as P

# %% [markdown]
# ## 1. Linear maps — `apply_matrix`
#
# A 2×2 matrix warps the coordinate grid to a lattice; the image basis arrows are its columns. The
# signed area of the image of the unit square is the **determinant** — we check it.

# %%
M = np.array([[1.0, 1.0], [0.0, 1.0]])  # a shear (det = 1)
mv.Plane(extent=3, grid=False).apply_matrix(M).display()

# image of the unit square, area via the shoelace formula, equals det M
img = maps.from_matrix(M)(mv.UNIT_SQUARE)
x, y = img[:, 0], img[:, 1]
area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
assert np.isclose(area, abs(np.linalg.det(M)))
print(f"shear: image-square area {area:.3f} == |det M| {abs(np.linalg.det(M)):.3f} ✓")

# a scaling doubles area (det = 4)
S = np.array([[2.0, 0.0], [0.0, 2.0]])
mv.Plane(extent=3, grid=False).apply_matrix(S, color=mv.GREEN).display()
assert np.isclose(np.linalg.det(S), 4.0)
print("uniform scale ×2: det = 4 (area ×4)")

# %% [markdown]
# ## 2. Geometric operators — `show_operator`
#
# `show_operator` overlays a faint domain grid with its bold image. A **projection** collapses the
# plane onto a line (non-invertible, det 0); a **rotation** preserves lengths.
#
# Pass `probe_shape=` to carry a recognizable figure along — here `mv.FLAG`, an asymmetric "flag on a
# pole" whose chirality makes flips and shears legible (`mv.UNIT_SQUARE`, or any array of points,
# work too; the default is `None`, i.e. grid only). `probe_shape_color=` colors its image separately
# from the grid's.


# %%
def projection_onto(a):
    a = np.asarray(a, float)
    return lambda pts: (
        (np.asarray(pts, float).reshape(-1, 2) @ a / (a @ a))[:, None] * a
    )


mv.Plane(extent=3, grid=False).show_operator(
    projection_onto([1, 0.5]),
    color=mv.ORANGE,
    probe_shape=mv.FLAG,
    probe_shape_color=mv.GREEN,
).display()
# projection is idempotent: P(P x) == P x
Pm = projection_onto([1, 0.5])
x = np.array([[1.3, -0.7]])
assert np.allclose(Pm(Pm(x)), Pm(x))
print("projection is idempotent ✓  (the flag collapses onto the line)")

theta = np.deg2rad(40)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
mv.Plane(extent=3, grid=False).show_operator(
    maps.from_matrix(R), color=mv.BLUE, probe_shape=mv.FLAG
).display()
assert np.allclose(R.T @ R, np.eye(2))  # orthogonal ⇒ preserves lengths
print("rotation is orthogonal ✓")

# %% [markdown]
# ## 3. Complex functions — `apply_complex` (conformal maps)
#
# A complex function bends the grid into curves while preserving angles (where analytic). The classic
# images: `z²` → confocal parabolas, `1/z` → circles (with a clean break at the pole).

# %%
mv.Plane(extent=2, grid=False).apply_complex(
    lambda z: z**2, color=mv.GREEN, step=0.5
).display()
mv.Plane(extent=2, grid=False).apply_complex(
    lambda z: 1 / z, color=mv.ORANGE, step=0.5
).display()

# conformality: a tiny circle maps to a near-circle under an analytic map
z0, r = 0.6 + 0.5j, 1e-2
c = z0 + r * np.exp(1j * np.linspace(0, 2 * np.pi, 200, endpoint=False))
w = c**2
d = np.abs(w - w.mean())
assert (d.max() - d.min()) / d.mean() < 0.1
print("z² is conformal (small circle → near-circle) ✓")

# the pole of 1/z produces a NaN that breaks the curve rather than streaking to infinity
line_through_0 = np.c_[np.linspace(-2, 2, 101), np.zeros(101)]
image = maps.from_complex(lambda z: 1 / z)(line_through_0)
assert np.isnan(image).any()
assert len(maps._finite_runs(image)) == 2  # split into two rays either side of the pole
print("1/z: pole handled — curve splits into 2 finite runs ✓")

# %% [markdown]
# ## 4. A Möbius transformation
#
# Möbius maps `(az+b)/(cz+d)` send circles-and-lines to circles-and-lines — the grid becomes a family
# of circles. They are the conformal automorphisms behind hyperbolic geometry and the Riemann sphere.


# %%
def mobius(a, b, c, d):
    return lambda z: (a * z + b) / (c * z + d)


f = mobius(1, -1j, 1, 1j)  # maps the real line / grid to circles
mv.Plane(extent=3, grid=False).apply_complex(f, color=mv.PURPLE, step=0.5).display()
# a Möbius map is invertible: composing with its inverse is the identity
finv = mobius(1j, 1j, -1, 1)  # inverse of the above (up to scale)
z = 0.3 + 0.7j
assert np.isclose(finv(f(z)), z)
print("Möbius map ∘ inverse = identity ✓")

# %% [markdown]
# **Recap.** `apply_matrix`, `show_operator`, and `apply_complex` are one push-forward, and every
# claim (determinant = area, idempotent projection, orthogonal rotation, conformality, pole splitting,
# Möbius invertibility) is checked numerically in the cells above.
