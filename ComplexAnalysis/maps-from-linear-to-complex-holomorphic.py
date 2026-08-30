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
mv.Plane(extent=3, grid=False).apply_matrix(M, color=mv.PURPLE).display()

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
    color=mv.PURPLE,
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
    lambda z: z**2, color=mv.PURPLE, step=0.5
).display()
mv.Plane(extent=2, grid=False).apply_complex(
    lambda z: 1 / z, color=mv.GREEN, step=0.5
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
# ## 5. Non-holomorphic maps — when the grid is *not* conformal
#
# `apply_complex` accepts **any** `g: ℂ→ℂ`, holomorphic or not. A holomorphic (conformal) map keeps
# the grid's angles: a little square maps to a (rotated, scaled) little square. A **non-holomorphic**
# map — one that depends on `z̄`, or otherwise fails to be angle-preserving — distorts the grid
# instead. Two extremes, both pushed through the same `push` machinery:
#
# * **`z̄·z = |z|²`** is *real-valued*, so its image collapses onto the real axis — a rank-1 map, the
#   most degenerate non-conformal case: the whole mesh folds onto a line.
# * **the twist** `f(x, y) = (r, θ + k·r)` stays 2D yet is still non-conformal — it leaves the radius
#   untouched (so circles map to circles) but is not angle-preserving, so a square curves into a
#   sheared quadrilateral. The grid **spirals**: area is preserved (det J = 1) while angles are not.

# %%
# (a) the degenerate case: f(z) = z̄·z = |z|² is real-valued ⇒ the grid collapses onto the real axis
mv.Plane(extent=2, grid=False).apply_complex(
    lambda z: np.conj(z) * z, color=mv.PURPLE, step=0.5
).display()

# check: the image is real-valued (imag ≈ 0), and a small circle collapses to a flat segment
g = maps.from_complex(lambda z: np.conj(z) * z)
xs = np.linspace(-2, 2, 11)
pts = np.c_[np.repeat(xs, len(xs)), np.tile(xs, len(xs))]
W = g(pts)
assert np.allclose(W[:, 1], 0.0, atol=1e-9)  # imag part ≈ 0 ⇒ on the real axis
z0, r = 0.6 + 0.5j, 1e-2
c = z0 + r * np.exp(1j * np.linspace(0, 2 * np.pi, 200, endpoint=False))
wc = g(np.c_[np.real(c), np.imag(c)])
assert wc[:, 1].max() - wc[:, 1].min() < 1e-9 and wc[:, 0].max() - wc[:, 0].min() > 1e-6
print(
    "z̄·z = |z|²: real-valued ⇒ the grid collapses onto the real axis (imag ≈ 0; a circle → a flat segment) ✓"
)

# %% [markdown]
# ### Measuring non-conformality: the "similarity deviation"
#
# Cells (b) and (c) certify non-holomorphic maps are non-conformal with a single scalar,
# `sim_dev(J)`. It measures how far a map's **Jacobian** `J` at a point `p` is from being a
# **similarity** — a scaled rotation `c·R` — because a 2-D map is conformal at `p` *iff* its
# Jacobian there is a similarity (this is the Cauchy–Riemann condition written in "real" terms).
#
# `jac(F, p)` returns `J`, a 2×2 matrix; its **rows** `r1 = J[0]` and `r2 = J[1]` are the **images of
# the domain axes** `e1`, `e2`. A similarity has exactly two hallmarks, and `sim_dev` checks both:
#
# ```python
# def sim_dev(J):
#     r1, r2 = J[0], J[1]                    # images of e1, e2
#     return abs(np.dot(r1, r2))            # 0 ⇔ right angle preserved (orthogonal)
#          + abs(np.linalg.norm(r1)
#             - np.linalg.norm(r2))         # 0 ⇔ no directional stretch (equal length)
# ```
#
# * **term 1** `|r1·r2|` — 0 when the image axes stay orthogonal (a 90° angle survives); nonzero means
#   angles are distorted.
# * **term 2** `||r1| − |r2||` — 0 when both image axes have equal length (no direction stretched more
#   than the other); nonzero means anisotropic stretching.
#
# So `sim_dev = 0` ⇔ the Jacobian is a scaled rotation ⇔ **conformal**; large ⇔ **non-conformal**.
# For `z²`, `J = [[1.4, 1], [−1, 1.4]]` gives `r1·r2 = 0` and `|r1| = |r2| = √2.96`, so `sim_dev = 0`
# (conformal). Both non-holomorphic maps below score large — the shear's `J = [[1, 2y], [0, 1]]` has
# `r1·r2 = 2y` and the twist's `J` has `r1·r2 ≈ 15.6` (a right angle badly wrecked) — *even though*
# both are area-preserving (`det J = 1`): the signature of a non-holomorphic map.
#
# **Caveat.** This is a *pointwise* (infinitesimal) test — it decides whether a map is conformal *at*
# `p`, not whether the image of a *finite* square stays a square: for a nonlinear map a finite square's
# corner angles drift even under a conformal map like `z²`. That is why we test the Jacobian, not a
# finite square.


# %%
# (b) the smooth shear f(x, y) = (x + y², y): a clean, *pure* non-conformality. It is a horizontal
# shear of strength 2y — area is preserved (det = 1) — but a right angle is distorted, so a square
# becomes a parallelogram. Horizontal grid lines stay horizontal; vertical lines bow into parabolas.
def shear(pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return np.c_[x + y * y, y]


mv.Plane(extent=2, grid=False).show_operator(shear, color=mv.GREEN, step=0.5).display()


# A conformal map's Jacobian is a scaled rotation (orthogonal, equal-length rows); the
# "similarity deviation" below is ≈ 0 for z² and large for any non-conformal map.
def jac(F, p, h=1e-5):
    p = np.asarray(p, float).reshape(1, 2)
    fx = (F(p + h * np.array([1.0, 0.0]))[0] - F(p - h * np.array([1.0, 0.0]))[0]) / (
        2 * h
    )
    fy = (F(p + h * np.array([0.0, 1.0]))[0] - F(p - h * np.array([0.0, 1.0]))[0]) / (
        2 * h
    )
    return np.stack([fx, fy])  # rows = images of e₁, e₂


def sim_dev(J):
    r1, r2 = J[0], J[1]
    return abs(np.dot(r1, r2)) + abs(np.linalg.norm(r1) - np.linalg.norm(r2))


# z² is conformal (dev ≈ 0); the shear preserves area (det = 1) but is non-conformal (dev ≫ 0)
assert (
    sim_dev(jac(lambda p: maps.from_complex(lambda z: z**2)(p), np.array([0.7, 0.5])))
    < 0.05
)
J = jac(shear, np.array([0.7, 0.5]))
assert np.isclose(np.linalg.det(J), 1.0, atol=1e-2) and sim_dev(J) > 1.0
print(
    f"shear (x + y², y): area-preserving (det J = {np.linalg.det(J):.2f}) but non-conformal (sim-dev {sim_dev(J):.2f} ≫ 0) ✓"
)


# %%
# (c) the twist f(x, y) = (r, θ + k·r): a *prettier* non-holomorphic map that does NOT collapse.
# It leaves the radius untouched (circles stay circles) but is not angle-preserving, so a square
# curves into a sheared quadrilateral — the grid spirals.
def twist(pts, k=1.5):
    pts = np.asarray(pts, float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    r = np.hypot(x, y)
    th = np.arctan2(y, x) + k * r
    return np.c_[r * np.cos(th), r * np.sin(th)]


mv.Plane(extent=2, grid=False).show_operator(twist, color=mv.BLUE, step=0.5).display()


# reusing jac / sim_dev from cell (b): the twist is area-preserving (det = 1) but non-conformal
J = jac(twist, np.array([0.7, 0.5]))
assert np.isclose(np.linalg.det(J), 1.0, atol=1e-2) and sim_dev(J) > 1.0
print(
    f"twist: area-preserving (det J = {np.linalg.det(J):.2f}) but non-conformal (sim-dev {sim_dev(J):.2f} ≫ 0) ✓"
)

# %% [markdown]
# **Recap.** `apply_matrix`, `show_operator`, and `apply_complex` are one push-forward. Every claim is
# checked numerically above: determinant = area, idempotent projection, orthogonal rotation,
# conformality, pole splitting, Möbius invertibility — and, for non-holomorphic maps, the *failure*
# of conformality: `z̄·z` collapses onto the real axis, while the shear and the twist keep area but
# shear the angles.

# %% [markdown]
#

# %% [markdown]
#
