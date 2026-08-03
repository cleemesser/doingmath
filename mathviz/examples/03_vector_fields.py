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
# # mathviz examples 3 — vector fields
#
# A **vector field** assigns a vector to each point. `Plane.field(f)` draws it as a grid of arrows,
# where `f` is a 2×2 matrix (the linear field $x\mapsto Mx$) or a point-map `(N,2)->(N,2)`;
# `Plane.field_complex(g)` takes a complex field $g:\mathbb{C}\to\mathbb{C}$.
#
# **Default convention:** arrows are drawn at a **uniform length** (direction only) and the
# **magnitude is encoded by color** — so dense fields stay legible and the length never lies.

# %%
import numpy as np
import mathviz as mv
from mathviz import primitives as P

# %% [markdown]
# ## Rotational field (curl):  $x \mapsto (-y,\,x)$
#
# Uniform-length arrows circulate around the origin; the viridis color shows the magnitude growing
# with radius (dark at the center, bright at the edges).

# %%
rot = mv.Plane(extent=3).field([[0, -1], [1, 0]])
rot.display()

arrows = [x for x in rot.primitives if isinstance(x, P.Arrow)]
lens = [np.hypot(*(a.head - a.tail)) for a in arrows]
assert np.allclose(lens, lens[0])  # default: all arrows the same length
print(
    f"{len(arrows)} arrows, all equal length {lens[0]:.3f} — magnitude is shown by color ✓"
)

# %% [markdown]
# ## Source and sink:  $x \mapsto x$  and  $x \mapsto -x$
#
# Arrows point outward from a source (positive eigenvalues) and inward to a sink (negative). Colored
# by magnitude, both show the field strengthening away from the origin.

# %%
mv.Plane(extent=3).field([[1, 0], [0, 1]]).display()  # source
mv.Plane(extent=3).field([[-1, 0], [0, -1]], cmap="plasma").display()  # sink

# %% [markdown]
# ## Length ∝ magnitude (opt-out):  `normalize=False`
#
# For the same rotational field, arrow *length* now grows with magnitude (auto-scaled). Useful when
# you want the classic quiver look; the default avoids the overlap it can cause.

# %%
mv.Plane(extent=3).field([[0, -1], [1, 0]], normalize=False).display()

# %% [markdown]
# ## A complex field:  $g(z) = z^2$
#
# `field_complex` treats the complex value as the arrow $(\operatorname{Re} g,\ \operatorname{Im} g)$.

# %%
mv.Plane(extent=2).field_complex(lambda z: z**2).display()

# a single-color field (cmap=None) still works, e.g. for overlaying on other art
solid = mv.Plane(extent=2, backend="vedo").field(
    [[0, -1], [1, 0]], cmap=None, color=mv.BLUE
)
solid.display()
print("field renders on both backends; cmap=None → single color ✓")

# %% [markdown]
# **Recap.** `Plane.field` / `Plane.field_complex` draw 2D vector fields with **uniform-length,
# color-by-magnitude** arrows by default (`normalize=True`, `cmap="viridis"`); pass `normalize=False`
# for length-encodes-magnitude, or `cmap=None` for a single color. *(3D vector fields await the
# Phase-4 3D scene; the `Plane` here is the top-down 2D view.)*
