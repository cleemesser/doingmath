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
# # from linear maps of the plane to analytic functions and beyond
#
# A re-run of `maps-from-linear-to-complex-holomorphic.py` in which every map is **animated**: a
# single parameter `t ∈ [0, 1]` and a builder `build(t) -> Plane`. `mathviz.animate` pre-renders the
# frames offscreen to PNG bytes, then an ipywidgets `Play`/slider swaps the cached images — instant
# scrubbing, and nothing depends on a live 3D notebook backend.
#
# > **Run this notebook live.** The scrubbers below are ipywidgets; they render in a running Jupyter
# > frontend (or VS Code), not in a static export. The GIF and the numeric checks show up anywhere.
#
# **Lie-group first.** Where a map is a group element we animate along its **one-parameter subgroup**
# `M(t) = exp(t·log M)` — the geodesic path (`kind="geodesic"` for `GL⁺(2,ℝ)`, `mobius_path(A)` for
# `PSL(2,ℂ)`). The geodesic never degenerates, because `det exp(A) = e^{tr A} > 0` always. Where the
# Lie-group path **cannot** reach the target — a projection (`det = 0`, no logarithm), or `z²`/`1/z`
# and the non-holomorphic maps (not Möbius) — we fall back to `lerp` (a straight line in matrix space)
# or the holomorphic `homotopy` (a convex combination), and the degeneration is the mathematics rather
# than an artifact.

# %%
import numpy as np
import mathviz as mv
from mathviz import maps
from mathviz.animate import (
    animate_matrix,
    matrix_path,
    homotopy,
    mobius_path,
    classify_mobius,
    scrubber,
    to_gif,
)
import tempfile
from pathlib import Path
from IPython.display import Image as IPyImage

SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])   # det = 1
SCALE = np.array([[2.0, 0.0], [0.0, 2.0]])   # det = 4
ROT = maps.rotation(np.deg2rad(60.0))        # det = +1
PROJ = maps.projection(np.pi / 6)            # det = 0 (singular)

# %% [markdown]
# ## 1. Linear maps — the geodesic one-parameter subgroup
#
# `animate_matrix(M, kind="geodesic")` scrubs `M(t) = exp(t·log M)`, the one-parameter subgroup of
# `GL⁺(2,ℝ)` through `M` — the Lie-group path. The view is pinned to `±extent`, so the grid warps
# without the camera rescaling. The shear is a pure `gl(2,ℝ)` element, so its geodesic is the shear
# flow `[[1, t], [0, 1]]`; a uniform scale flows to `t·c·I`.

# %%
mv.animate_matrix(SHEAR, kind="geodesic", extent=3, size=(460, 460), width=540, height=540, n=24)

# %%
mv.animate_matrix(SCALE, kind="geodesic", extent=3, size=(460, 460), width=540, height=540, n=24)


# %%

# the geodesic is a one-parameter subgroup: M(s)·M(t) = M(s+t), and det M(t) = exp(t·tr L) > 0
for M in (SHEAR, SCALE):
    at = matrix_path(M, "geodesic")
    print(
        f"{np.round(M, 1)}  subgroup M(s)M(t)==M(s+t)? {np.allclose(at(0.3) @ at(0.4), at(0.7))}  "
        f"det always > 0? {all(np.linalg.det(at(t)) > 0 for t in np.linspace(0, 1, 11))}"
    )

# %% [markdown]
# ## 2. Operators — rotation is a geodesic; projection cannot be
#
# A **rotation** lies in `SO(2)`, and its geodesic is a pure rotation (`det = 1` throughout, never
# degenerates). A **projection** has `det = 0`: it sits on the `det = 0` wall, so it has **no
# logarithm at all** — no one-parameter subgroup reaches it. Only `lerp` (a straight line in matrix
# space) gets there, and it does so by *collapsing* through the zero matrix at `t = 1/2`.

# %%
mv.animate_matrix(ROT, kind="geodesic", extent=3, size=(460, 460), width=500, height=500,
                   n=24)


# %%

# rotation: the geodesic det stays 1 for every t
print(
    "rotation geodesic det == 1 throughout?",
    np.allclose([np.linalg.det(matrix_path(ROT, "geodesic")(t)) for t in np.linspace(0, 1, 11)], 1.0),
)

# %% [markdown]
# A **projection** is singular (`det = 0`), so no real logarithm exists — `geodesic`/`polar` refuse.
# `lerp` is the only path that reaches it, and the collapse you see at `t = 1/2` is the mathematics,
# not an artifact.

# %%
mv.animate_matrix(PROJ, kind="lerp", extent=3, size=(460, 460), width=340, height=340, n=24, ping_pong=False)


# %%

# the projection is rank 1 (det = 0): the only reachable path is lerp, and it collapses through det 0
print("projection rank 1 (det = 0):", np.linalg.matrix_rank(PROJ), "  det =", round(np.linalg.det(PROJ), 3))

# %% [markdown]
# ## 3. Complex functions — `z²` and `1/z` are *not* Möbius, so no geodesic
#
# The Lie-group path for complex maps is the `PSL(2,ℂ)` one-parameter subgroup (`mobius_path`), but
# that only covers **Möbius** maps `(az+b)/(cz+d)` — degree 1, a single pole. `z²` (degree 2) and
# `1/z` (a pole at 0) are **not** Möbius, so no geodesic reaches them. We fall back to the holomorphic
# `homotopy` `g_t = (1-t)·z + t·g(z)`: a convex combination of holomorphic maps, so every frame is a
# genuine holomorphic map; conformality, however, can break where `g_t'` vanishes.

# %%
mv.scrubber(
    lambda t: mv.Plane(extent=6, grid=False, size=(460, 460)).apply_complex(homotopy(lambda z: z**2)(t), color=mv.GREEN, step=0.5),
    n=30,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)


# %%

# the homotopy is holomorphic at every t and reaches the target at t = 1 (identity at t = 0)
g = homotopy(lambda z: z**2)
z = 0.5 + 0.5j
print("homotopy z²:  g_0(z)==z?", np.isclose(g(0)(z), z), "  g_1(z)==z²?", np.isclose(g(1)(z), z**2))

# %%
mv.scrubber(
    lambda t: mv.Plane(extent=3, grid=False, size=(460, 460)).apply_complex(homotopy(lambda z: 1 / z)(t), color=mv.ORANGE, step=0.5),
    n=18,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)

# %% [markdown]
# ## 4. Möbius — the `PSL(2,ℂ)` geodesic
#
# Möbius maps `(az+b)/(cz+d)` **are** a group — the conformal automorphisms of the Riemann sphere,
# `PSL(2,ℂ)` — so the Lie-group path applies. `mobius_path(A)` is the one-parameter subgroup
# `t ↦` the Möbius map of `exp(t·log A)`, the complex mirror of `geodesic_path`. The original
# `f = (z − i)/(z + i)` has matrix `A = [[1, −i], [1, i]]`.

# %%
A = np.array([[1.0, -1j], [1.0, 1j]])
mv.scrubber(
    lambda t: mv.Plane(extent=3, grid=False, size=(460, 460)).apply_complex(mobius_path(A)(t), color=mv.PURPLE, step=0.5),
    n=18,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)


# %%

# mobius_path(A) reaches the Möbius map of A at t = 1 (the map is invariant under scaling A);
# classify_mobius sorts the subgroup into the real/complex family
z = 0.3 + 0.7j
print("mobius_path(A)(1) == (z−i)/(z+i)?", np.isclose(mobius_path(A)(1)(z), (z - 1j) / (z + 1j)))
print("classify_mobius(A):", classify_mobius(A))

# %% [markdown]
# ## 5. Non-holomorphic maps — the geodesic does not reach them either
#
# None of these is a Möbius map, so no `PSL(2,ℂ)` geodesic applies. We fall back to the straight-line
# sweep (the holomorphic `homotopy` where it exists, and a direct parameter sweep for the point-maps):
#
# * **`z̄·z = |z|²`** — the `homotopy` to a real-valued map; the grid collapses onto the real axis.
# * **the shear `f(x,y) = (x + y², y)`** — a non-linear point map; a straight-line sweep of its
#   strength from 0 to 1 (no matrix, hence no geodesic).
# * **the twist `f(x,y) = (r, θ + k·r)`** — a straight-line sweep of its twist angle from 0 to k
#   (at `t = 0` the twist angle is 0, i.e. the identity).

# %%
# (a) z̄·z = |z|² via the homotopy: the grid morphs to the collapse onto the real axis
mv.scrubber(
    lambda t: mv.Plane(extent=2, grid=False, size=(460, 460)).apply_complex(homotopy(lambda z: np.conj(z) * z)(t), color=mv.RED, step=0.5),
    n=18,
    width=340,
    height=340,
    label="t",
    fmt="{:.2f}",
)


# %%
# at t = 1 the image is real-valued (imag ≈ 0) — the collapse, reached by a straight-line sweep
g = homotopy(lambda z: np.conj(z) * z)
print("homotopy z̄·z: imag part at t=1 ≈ 0?", np.allclose(np.imag(g(1)(np.linspace(-2, 2, 9) + 0.5j)), 0.0, atol=1e-9))


# %%
# (b) the shear: a straight-line sweep of its strength (non-linear ⇒ no geodesic)
def shear_t(t):
    return lambda pts: np.c_[pts[:, 0] + t * pts[:, 1] ** 2, pts[:, 1]]


mv.scrubber(
    lambda t: mv.Plane(extent=2, grid=False, size=(460, 460)).show_operator(shear_t(t), color=mv.GREEN, step=0.5),
    n=18,
    width=340,
    height=340,
    label="t",
    fmt="{:.2f}",
)


# %%
# at t = 0 the map is the identity; at t = 1 it is the shear
print("shear t=0 is identity?", np.allclose(shear_t(0)(np.array([[0.3, -0.7]])), [[0.3, -0.7]]))


# %%
# (c) the twist: a straight-line sweep of its twist angle (t=0 ⇒ no twist ⇒ the identity)
def twist_t(t, k=1.5):
    def f(pts):
        x, y = pts[:, 0], pts[:, 1]
        r = np.hypot(x, y)
        th = np.arctan2(y, x) + t * k * r
        return np.c_[r * np.cos(th), r * np.sin(th)]

    return f


mv.scrubber(
    lambda t: mv.Plane(extent=2, grid=False, size=(460, 460)).show_operator(twist_t(t), color=mv.BLUE, step=0.5),
    n=18,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)


# %%
# at t = 0 the twist angle is 0, so the map is the identity
print("twist t=0 is identity?", np.allclose(twist_t(0)(np.array([[0.6, 0.5]])), [[0.6, 0.5]]))

# %% [markdown]
# ## Portable GIF export
#
# The same builder, written to an animated GIF — portable, renders on GitHub, needs no widget stack.
# Here the geodesic shear flow.

# %%
out = Path(tempfile.mkdtemp()) / "shear-geodesic.gif"
mv.to_gif(
    lambda t: mv.Plane(extent=3, grid=False, size=(360, 360)).push(
        matrix_path(SHEAR, "geodesic")(t), probe_shape=mv.UNIT_SQUARE, basis=True, samples=2
    ),
    out,
    n=20,
    fps=15,
    ping_pong=True,
)
IPyImage(filename=str(out))

# %% [markdown]
# **Recap.** The Lie-group path (the geodesic / `mobius_path`) is the principled one for maps that
# *are* group elements: the one-parameter subgroup `exp(t·log M)` (and `PSL(2,ℂ)` for Möbius) flows
# along the Lie algebra element that generates the map, and `det exp(A) = e^{tr A} > 0` means it
# never degenerates. Where it cannot reach the target — a projection (`det = 0`, no logarithm), or
# `z²`/`1/z` and the non-holomorphic maps (not Möbius) — we fall back to `lerp` or the holomorphic
# `homotopy`, and the degeneration / straight-line is the mathematics, not an artifact.

# %%
