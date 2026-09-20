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
# This notebook re-runs `maps-from-linear-to-complex-holomorphic.py` with one change: every map
# is animated. One parameter `t ∈ [0, 1]` and a builder `build(t) -> Plane` define each animation.
# `clmmathtools.viz.animate` renders the frames in advance, offscreen, to PNG bytes. An ipywidgets
# `Play` button and slider then swap the cached images. Scrubbing is instant, and nothing depends
# on a live 3D notebook backend.
#
# ### Todo
# - [ ] Update the text: move my notes on the animation technique to another document. I do not want to introduce Lie groups here.
# - [ ] Update the text for learners. The learner will use this notebook after an intro to linear algebra and transformations, and after an intro to the complex plane.
# - [ ] Give the scrubber a standard size, probably about 400 x 400. Render the pixels at a higher resolution, and increase the extent to add more squares with thinner lines.
# - [x] Use purple as the default color, with green or blue for variety
#
# > Run this notebook live. The scrubbers below are ipywidgets. They render in a running Jupyter
# > frontend or VS Code, but not in a static export. The GIF and the numeric checks show up
# > anywhere.
#
# Lie-group first: where a map is a group element, animate along its one-parameter subgroup
# `M(t) = exp(t·log M)`. This is the geodesic path (`kind="geodesic"` for `GL⁺(2,ℝ)`,
# `mobius_path(A)` for `PSL(2,ℂ)`). The geodesic never degenerates, because
# `det exp(A) = e^{tr A} > 0` always holds. Where the Lie-group path cannot reach the target, fall
# back to `lerp` (a straight line in matrix space) or the holomorphic `homotopy` (a convex
# combination). The path cannot reach a projection (`det = 0`, no logarithm), or `z²`, `1/z`, and
# the non-holomorphic maps (they are not Möbius). In those cases the degeneration is the
# mathematics, not an artifact.

# %% jupyter={"source_hidden": true}
import numpy as np
import clmmathtools.viz as mv
from clmmathtools.viz import maps
from clmmathtools.viz.animate import (
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

SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])  # det = 1
SCALE = np.array([[2.0, 0.0], [0.0, 2.0]])  # det = 4
ROT = maps.rotation(np.deg2rad(60.0))  # det = +1
PROJ = maps.projection(np.pi / 6)  # det = 0 (singular)

# %% [markdown]
# ## 1. Linear maps: the geodesic one-parameter subgroup
#
# `animate_matrix(M, kind="geodesic")` scrubs `M(t) = exp(t·log M)`. This is the one-parameter
# subgroup of `GL⁺(2,ℝ)` through `M`, that is, the Lie-group path. The view is fixed to `±extent`,
# so the grid warps while the camera keeps its scale. The shear is a pure `gl(2,ℝ)` element, so
# its geodesic is the shear flow `[[1, t], [0, 1]]`. A uniform scale flows to `t·c·I`.

# %% jupyter={"source_hidden": true}
mv.animate_matrix(
    SHEAR, kind="geodesic", extent=3, size=(460, 460), width=400, height=400, n=24
)

# %% jupyter={"source_hidden": true}
mv.animate_matrix(SCALE, kind="geodesic", color=mv.GREEN, extent=3, width=540, height=540, n=24)


# %% jupyter={"source_hidden": true}

# the geodesic is a one-parameter subgroup: M(s)·M(t) = M(s+t), and det M(t) = exp(t·tr L) > 0
for M in (SHEAR, SCALE):
    at = matrix_path(M, "geodesic")
    print(
        f"{np.round(M, 1)}  subgroup M(s)M(t)==M(s+t)? {np.allclose(at(0.3) @ at(0.4), at(0.7))}  "
        f"det always > 0? {all(np.linalg.det(at(t)) > 0 for t in np.linspace(0, 1, 11))}"
    )

# %% [markdown]
# ## 2. Operators: rotation has a geodesic, projection does not
#
# A rotation lies in `SO(2)`, and its geodesic is a pure rotation. Its `det` stays 1, so it never
# degenerates. A projection has `det = 0`. It sits on the `det = 0` wall, so it has no logarithm
# at all, and no one-parameter subgroup reaches it. Only `lerp` (a straight line in matrix space)
# gets there. It arrives by collapsing through the zero matrix at `t = 1/2`.

# %% jupyter={"source_hidden": true}
mv.animate_matrix(
    ROT, kind="geodesic", color=mv.BLUE, extent=3, size=(460, 460), width=500, height=500, n=24
)


# %% jupyter={"source_hidden": true}

# rotation: the geodesic det stays 1 for every t
print(
    "rotation geodesic det == 1 throughout?",
    np.allclose(
        [np.linalg.det(matrix_path(ROT, "geodesic")(t)) for t in np.linspace(0, 1, 11)],
        1.0,
    ),
)

# %% [markdown]
# A projection is singular (`det = 0`), so no real logarithm exists, and `geodesic` and `polar`
# refuse it. `lerp` is the only path that reaches it. The collapse you see at `t = 1/2` is the
# mathematics, not an artifact.

# %% jupyter={"source_hidden": true}
mv.animate_matrix(
    PROJ,
    kind="lerp",
    extent=3,
    size=(460, 460),
    width=340,
    height=340,
    n=24,
    ping_pong=False,
)


# %% jupyter={"source_hidden": true}

# the projection is rank 1 (det = 0): the only reachable path is lerp, and it collapses through det 0
print(
    "projection rank 1 (det = 0):",
    np.linalg.matrix_rank(PROJ),
    "  det =",
    round(np.linalg.det(PROJ), 3),
)

# %% [markdown]
# ## 3. Complex functions: `z²` and `1/z` are not Möbius, so no geodesic
#
# The Lie-group path for complex maps is the `PSL(2,ℂ)` one-parameter subgroup (`mobius_path`).
# It covers only Möbius maps `(az+b)/(cz+d)`, which have degree 1 and a single pole. `z²` has
# degree 2, and `1/z` has a pole at 0, so neither is a Möbius map, and no geodesic reaches them.
# Fall back to the holomorphic `homotopy` `g_t = (1-t)·z + t·g(z)`. This is a convex combination
# of holomorphic maps (a weighted mix), so every frame is a genuine holomorphic map. Even so,
# conformality can break where `g_t'` is 0.

# %% jupyter={"source_hidden": true}
mv.scrubber(
    lambda t: mv.Plane(extent=6, grid=False, size=(800, 800)).apply_complex(
        homotopy(lambda z: z**2)(t), color=mv.PURPLE, step=0.5
    ),
    n=30,
    width=400,
    height=400,
    label="t",
    fmt="{:.2f}",
)


# %% jupyter={"source_hidden": true}

# the homotopy is holomorphic at every t and reaches the target at t = 1 (identity at t = 0)
g = homotopy(lambda z: z**2)
z = 0.5 + 0.5j
print(
    "homotopy z²:  g_0(z)==z?",
    np.isclose(g(0)(z), z),
    "  g_1(z)==z²?",
    np.isclose(g(1)(z), z**2),
)

# %% jupyter={"source_hidden": true}
mv.scrubber(
    lambda t: mv.Plane(extent=3, grid=False, size=(460, 460)).apply_complex(
        homotopy(lambda z: 1 / z)(t), color=mv.GREEN, step=0.5
    ),
    n=18,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)

# %% [markdown]
# ## 4. Möbius maps: the `PSL(2,ℂ)` geodesic
#
# Möbius maps `(az+b)/(cz+d)` form a group: the invertible conformal maps of the Riemann sphere,
# `PSL(2,ℂ)`. So the Lie-group path applies. `mobius_path(A)` is the one-parameter subgroup that
# sends `t` to the Möbius map of `exp(t·log A)`. This is the complex counterpart of
# `geodesic_path`. The map `f = (z − i)/(z + i)` has matrix `A = [[1, −i], [1, i]]`.

# %% jupyter={"source_hidden": true}
A = np.array([[1.0, -1j], [1.0, 1j]])
mv.scrubber(
    lambda t: mv.Plane(extent=5, grid=False, size=(460, 460)).apply_complex(
        mobius_path(A)(t), color=mv.PURPLE, step=0.5
    ),
    n=18,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)


# %% jupyter={"source_hidden": true}

# mobius_path(A) reaches the Möbius map of A at t = 1 (the map is invariant under scaling A);
# classify_mobius sorts the subgroup into the real/complex family
z = 0.3 + 0.7j
print(
    "mobius_path(A)(1) == (z−i)/(z+i)?",
    np.isclose(mobius_path(A)(1)(z), (z - 1j) / (z + 1j)),
)
print("classify_mobius(A):", classify_mobius(A))

# %% [markdown]
# ## 5. Non-holomorphic maps: the geodesic does not reach them either
#
# None of these maps is a Möbius map, so no `PSL(2,ℂ)` geodesic applies. Fall back to a
# straight-line sweep: the holomorphic `homotopy` where it exists, or a direct parameter sweep
# for the point maps.
#
# * `z̄·z = |z|²`: the `homotopy` runs to a real-valued map, and the grid collapses onto the real
#   axis.
# * The shear `f(x,y) = (x + y², y)`: a non-linear point map. The sweep takes its strength from 0
#   to 1. There is no matrix, so there is no geodesic.
# * The twist `f(x,y) = (r, θ + k·r)`: the sweep takes its twist angle from 0 to k. At `t = 0`
#   the twist angle is 0, so the map is the identity.

# %% jupyter={"source_hidden": true}
# (a) z̄·z = |z|² via the homotopy: the grid morphs to the collapse onto the real axis
mv.scrubber(
    lambda t: mv.Plane(extent=2, grid=False, size=(460, 460)).apply_complex(
        homotopy(lambda z: np.conj(z) * z)(t), color=mv.PURPLE, step=0.5
    ),
    n=18,
    width=340,
    height=340,
    label="t",
    fmt="{:.2f}",
)


# %% jupyter={"source_hidden": true}
# at t = 1 the image is real-valued (imag ≈ 0) — the collapse, reached by a straight-line sweep
g = homotopy(lambda z: np.conj(z) * z)
print(
    "homotopy z̄·z: imag part at t=1 ≈ 0?",
    np.allclose(np.imag(g(1)(np.linspace(-2, 2, 9) + 0.5j)), 0.0, atol=1e-9),
)


# %% jupyter={"source_hidden": true}
# (b) the shear: a straight-line sweep of its strength (non-linear ⇒ no geodesic)
def shear_t(t):
    return lambda pts: np.c_[pts[:, 0] + t * pts[:, 1] ** 2, pts[:, 1]]


mv.scrubber(
    lambda t: mv.Plane(extent=2, grid=False, size=(460, 460)).show_operator(
        shear_t(t), color=mv.GREEN, step=0.5
    ),
    n=18,
    width=340,
    height=340,
    label="t",
    fmt="{:.2f}",
)


# %% jupyter={"source_hidden": true}
# at t = 0 the map is the identity; at t = 1 it is the shear
print(
    "shear t=0 is identity?",
    np.allclose(shear_t(0)(np.array([[0.3, -0.7]])), [[0.3, -0.7]]),
)


# %% jupyter={"source_hidden": true}
# (c) the twist: a straight-line sweep of its twist angle (t=0 ⇒ no twist ⇒ the identity)
def twist_t(t, k=1.5):
    def f(pts):
        x, y = pts[:, 0], pts[:, 1]
        r = np.hypot(x, y)
        th = np.arctan2(y, x) + t * k * r
        return np.c_[r * np.cos(th), r * np.sin(th)]

    return f


mv.scrubber(
    lambda t: mv.Plane(extent=5, grid=False, size=(460, 460)).show_operator(
        twist_t(t), color=mv.BLUE, step=0.5
    ),
    n=18,
    width=500,
    height=500,
    label="t",
    fmt="{:.2f}",
)


# %% jupyter={"source_hidden": true}
# at t = 0 the twist angle is 0, so the map is the identity
print(
    "twist t=0 is identity?",
    np.allclose(twist_t(0)(np.array([[0.6, 0.5]])), [[0.6, 0.5]]),
)

# %% [markdown]
# ## Portable GIF export
#
# The same builder can write an animated GIF. A GIF is portable, renders on GitHub, and needs no
# widget stack. The example here is the geodesic shear flow.

# %% jupyter={"source_hidden": true}
out = Path(tempfile.mkdtemp()) / "shear-geodesic.gif"
mv.to_gif(
    lambda t: mv.Plane(extent=3, grid=False, size=(360, 360)).push(
        matrix_path(SHEAR, "geodesic")(t),
        color=mv.PURPLE,
        probe_shape=mv.UNIT_SQUARE,
        basis=True,
        samples=2,
    ),
    out,
    n=20,
    fps=15,
    ping_pong=True,
)
IPyImage(filename=str(out))

# %% [markdown]
# The Lie-group path (the geodesic or `mobius_path`) is the principled choice for maps that are
# group elements. The one-parameter subgroup `exp(t·log M)` (and `PSL(2,ℂ)` for Möbius maps)
# flows along the Lie algebra element that generates the map. Because `det exp(A) = e^{tr A} > 0`,
# it never degenerates. Where it cannot reach the target, fall back to `lerp` or the holomorphic
# `homotopy`. This happens for a projection (`det = 0`, no logarithm), and for `z²`, `1/z`, and
# the non-holomorphic maps (they are not Möbius). In those cases the degeneration or the
# straight-line path is the mathematics, not an artifact.

# %% jupyter={"source_hidden": true}
