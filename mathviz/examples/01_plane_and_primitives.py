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
# # mathviz examples 1 — the Plane and its primitives
#
# `mathviz` gives you one `Plane` you draw on with chainable calls; a **backend** (matplotlib or
# vedo) turns the recorded primitives into pixels. This notebook demonstrates every base primitive
# and asserts, inline, that each call records what it should — so it doubles as a smoke test.

# %%
import numpy as np
import mathviz as mv
from mathviz import primitives as P

print("mathviz", mv.__version__)

# %% [markdown]
# ## Primitives on the default plane
#
# A fresh `Plane` starts with a coordinate **grid** and **axes**. Then we add a vector, the standard
# **basis**, a segment, a full **line** through the origin, a parametric **curve** (the unit circle),
# some **points**, and a **text** label.

# %%
t = np.linspace(0, 2 * np.pi, 200)
circle = np.c_[np.cos(t), np.sin(t)]

p = mv.Plane(extent=3)
(
    p.vector([2, 1], color=mv.BLUE, label="v")
    .basis()
    .segment([-2, -2], [2, -1], color=mv.PURPLE)
    .line([0, 0], [1, 2], color=mv.GREY)
    .curve(circle, color=mv.GREEN)
    .points([[2, 2], [-1, 1.5]], color=mv.ORANGE, size=9)
    .text([-2.6, 2.5], "primitives")
)
p.display()

# each call recorded exactly one primitive of the expected type
kinds = [type(x) for x in p.primitives]
assert kinds.count(P.Grid) == 1
assert kinds.count(P.Arrow) == 3  # v + basis(2)
assert kinds.count(P.Segment) == 2 + 1 + 1  # axes(2) + segment + line
assert kinds.count(P.Polyline) == 1  # the circle
assert kinds.count(P.Points) == 1
assert kinds.count(P.Text) == 1
print("primitive bookkeeping OK:", {k.__name__: kinds.count(k) for k in set(kinds)})

# %% [markdown]
# ## The scene is backend-independent
#
# The same drawing code produces the *same* primitive list regardless of backend — only rendering
# differs. Here is the identical scene under matplotlib and vedo.


# %%
def build(backend):
    q = mv.Plane(extent=3, backend=backend)
    q.basis().curve(circle, mv.BLUE).points([[2, 1]], mv.ORANGE).text(
        [-2.6, 2.5], backend
    )
    return q


mpl_plane, vedo_plane = build("mpl"), build("vedo")
assert [type(x) for x in mpl_plane.primitives] == [
    type(x) for x in vedo_plane.primitives
]
print("matplotlib and vedo build identical scenes ✓")

mpl_plane.display()
vedo_plane.display()

# %% [markdown]
# ## Rasters — the substrate for phase portraits (Phase 3 preview)
#
# `raster(rgb)` fills the plane with an image, over which vectors/curves compose. Here we color each
# point by `arg(z)` — a first, un-enhanced **phase portrait** of the identity `f(z)=z`, previewing the
# complex-analysis work to come. (The matplotlib backend is the raster path.)

# %%
import matplotlib.cm as cm

E, res = 3.0, 500
xs = np.linspace(-E, E, res)
X, Y = np.meshgrid(xs, xs)
Z = X + 1j * Y
hue = (np.angle(Z) / (2 * np.pi)) % 1.0  # phase → [0,1)
rgb = cm.hsv(hue)[..., :3]

pp = mv.Plane(extent=E, grid=False, axes=False)
pp.raster(rgb).grid(alpha=0.25).text([-2.7, 2.6], "arg(z)")
pp.display()

assert isinstance(pp.primitives[0], P.Raster)
assert pp.primitives[0].rgb.shape == (res, res, 3)
print("raster recorded:", pp.primitives[0].rgb.shape)

# %% [markdown]
# **Recap.** One `Plane`, chainable primitive calls, a backend-independent scene, and a raster layer
# for phase portraits. Notebook 2 pushes the plane through **maps**.
