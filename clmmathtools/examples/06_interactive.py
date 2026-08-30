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
# # clmmathtools examples 6 — interactive rendering
#
# By default `clmmathtools` renders **static PNGs** (so figures survive a headless `jupytext --execute` and
# show on GitHub). The **vedo** backend can also produce a **live, orbitable widget** — drag to rotate,
# scroll to zoom — which is where 3D really pays off.
#
# > **Run this notebook live.** The interactive cells below render as widgets only in a running Jupyter
# > frontend (Lab/Notebook). In a static export (GitHub, `jupytext --execute`) they won't show an image —
# > that's expected. The **static** cells embed a normal PNG for reference.
# Note that this runs best for me in a real jupyter notebook or jupyter lab notebook
#   - some features run inside a vscode jupyter window

# %%
import numpy as np
import clmmathtools.viz as mv

print("running inside a notebook kernel?  mv.in_notebook() =", mv.in_notebook())

# %% [markdown]
# ## Static vs. interactive — the same scene
#
# First the static default (embeds a PNG). Then the identical scene as a **live widget** via
# `display(interactive=True)` — orbit it to see the parallelepiped from any angle.

# %%
o = [0, 0, 0]
a, b, c = [2, 0.3, 0], [0.4, 1.8, 0.2], [0.3, 0.5, 1.6]

# static — a normal embedded image
mv.Space3D(bounds=3).parallelepiped(o, a, b, c, facecolor=mv.ORANGE).display()

# %%
# interactive — a live widget (drag to orbit). Static exports show nothing here; run it live.
mv.Space3D(bounds=3).parallelepiped(o, a, b, c, facecolor=mv.ORANGE).display(
    interactive=True
)

# %% [markdown]
# ## Choosing when to be interactive
#
# - `display(interactive=True/False)` — per call.
# - `mv.set_interactive(True | False | "auto")` — global default. **`"auto"`** is the ergonomic choice:
#   **live in a notebook, static when headless** (it consults `mv.in_notebook()`).
# - `mv.set_vedo_display("k3d" | "trame" | "ipyvtklink")` — pick vedo's live display backend.
#   `k3d` renders in JS and carries its own camera, so it starts from *its* default view; `trame`
#   keeps a live VTK renderer and so honors clmmathtools's z-up `(elev, azim)`.

# the vedo ipyvtklink backend si the most robust as it handles images correctly, but it opens a new window to display graphics and thus requires an active desktop display.
# the k3d backend works pretty well, but 2D images are not currently handled well
# TODO: add on image handling to the k3d code -- likely using textures to do so
#  see plt_texture = k3d.texture(binary=image_data, file_format='png')
#     plot = k3d.plot(); plot += plt_texture; plot.display()
# %%
mv.set_interactive("auto")  # live here in Jupyter, static under jupytext --execute
try:
    mv.Space3D(bounds=3).field(
        lambda P: np.c_[-P[:, 1], P[:, 0], 0.3 * P[:, 2]], n=6
    ).display()
finally:
    mv.set_interactive(False)  # restore the static default even if the render raises

# TODO: suppress INFO in helpers: "converting int64 array to in32 for JS compatibility"

# %% [markdown]
# ## An analytic landscape you can orbit
#
# The phase-colored surface `|f(z)|` is far easier to read when you can turn it. Static first, then live.

# %%
f = lambda z: (z**2 - 1) / (z**2 + 1)  # zeros at ±1, poles at ±i
mv.Space3D(bounds=2.5).landscape(f, res=120, zmax=3).display()  # static reference

# %%
mv.Space3D(bounds=2.5).landscape(f, res=120, zmax=3).display(
    interactive=True
)  # live — orbit it

# %% [markdown]
# ## 2D scenes can be interactive too
#
# The 2D plane is normally viewed top-down, but the vedo widget lets you tilt it — handy for seeing how
# a phase portrait sits in the plane.

# %%
mv.set_vedo_display(
    "ipyvtklink"
)  # this handles images while k3d backend does not currently
mv.Plane(extent=2, grid=False, axes=False, backend="vedo").phase_portrait(
    f, res=300
).display(interactive=True)
# will need to use textures to emmulate images in k3d I think - clm

# %% [markdown]
# ## Under the hood
#
# `display(interactive=True)` returns the **widget object** (which the notebook renders); the static
# path returns `None` (it pushes a PNG). We can check that without drawing:

# %%
widget = (
    mv.Space3D(bounds=1)
    .parallelepiped(o, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    .display(interactive=True)
)
print(
    "interactive returns a live widget:", type(widget).__name__, "→", widget is not None
)

# %% [markdown]
# **Recap.** Static PNG is the default (portable, GitHub-friendly); opt into a **live vedo widget** with
# `display(interactive=True)`, or set it globally with `mv.set_interactive("auto")` and choose the vedo
# display backend with `mv.set_vedo_display(...)`. Interactive 3D — landscapes, Riemann surfaces,
# parallelepipeds, vector fields — is where vedo shines over the static matplotlib backend.
