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
#

# %% [markdown]
# # clmmathtools examples 7 — animating linear transformations
#
# Every animation here has the same shape: a **single parameter** $t$ running $0 \to 1$, and a
# *builder* `build(t)` that draws the scene at that instant. `clmmathtools.viz.animate` pre-renders the frames
# offscreen to PNG bytes, then an ipywidgets `Play`/slider swaps the cached images — so scrubbing is
# instant no matter how expensive the scene is, and nothing depends on a live 3D notebook backend.
#
# > **Run this notebook live.** The scrubbers below are ipywidgets; they render in a running Jupyter
# > frontend (or VS Code), not in a static export. The GIF and the numeric checks show up either way.

# %%
import numpy as np
import sympy as sp
from scipy.linalg import expm, logm

import clmmathtools.viz as mv
from clmmathtools.viz.animate import geodesic_path, lerp_path, matrix_path, polar_path, real_logm

SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])  # det = 1
HALF_TURN = np.array([[-1.0, 0.0], [0.0, -1.0]])  # rotation by π
GROW = np.array([[1.5, 0.3], [0.0, 1.2]])  # det = 1.8, area-expanding

# %% [markdown]
# ## The headline: `animate_matrix`
#
# Faint domain grid, bold warped image, and the basis arrows $M(t)\,e_1$, $M(t)\,e_2$. The view is
# pinned to $\pm$`extent`, so the grid warps *without the camera rescaling* — the whole point.

# %%
mv.animate_matrix(SHEAR, extent=2, n=24)

# %% [markdown]
# ## Which path from $I$ to $M$?
#
# To animate we need a **path of matrices** $M(t)$ with $M(0) = I$ and $M(1) = M$. There is no
# single best choice for every transformation:
#
# 1. ** linear interpolation (`lerp`) ** — the straight line $M(t) = (1-t)I + tM$, drawn **in matrix space**. But matrix space
#    path will often not respect the underlying geometry.
# 2. **`polar` interpoation ** — split $M = R\,P$ into a rotation $R$ and a symmetric positive-definite stretch $P$
#    (the SVD polar decomposition), then turn $R$ through $t\theta$ and raise $P$ to the power $t$.
# 3. **`geodesic` interpoation ** — the **one-parameter subgroup** $M(t) = \exp(t \log M)$: flow for time $t$ along
#    the constant velocity $L = \log M$ in the Lie algebra $\mathfrak{gl}(2,\mathbb{R})$. This is the
#    `LieGroups/` machinery of this repo, applied to $\mathrm{GL}^{+}(2,\mathbb{R})$.
#
# The failure is easiest to see on the **half-turn** $M = -I$. Watch $\det M(t)$:

# %%
ts = np.linspace(0, 1, 5)
KINDS = ("lerp", "polar", "geodesic")

print("M = -I  (rotation by π)        det M(t)\n")
print(f"{'t':>6} " + " ".join(f"{k:>10}" for k in KINDS))
for t in ts:
    dets = [np.linalg.det(matrix_path(HALF_TURN, k)(t)) for k in KINDS]
    print(f"{t:6.2f} " + " ".join(f"{d:10.3f}" for d in dets))

# the lerp really does pass through the *zero matrix*
assert np.allclose(lerp_path(HALF_TURN)(0.5), np.zeros((2, 2)))
print(
    "\nlerp at t = 1/2 is the zero matrix — the plane collapses to a point and pops back."
)

# %% [markdown]
# So `lerp` sends $\det$ to $0$ at $t = \tfrac12$: the grid shrinks to nothing, then re-expands
# "inside out". `polar` and `geodesic` hold $\det = 1$ throughout — they *rotate*, which is what a
# half-turn actually is. Scrub all three and compare.

# %%
mv.animate_matrix(HALF_TURN, kind="lerp", n=24, ping_pong=False)  # collapses at t = 1/2

# %%
mv.animate_matrix(
    HALF_TURN, kind="polar", n=24, ping_pong=False
)  # rotates, never degenerates

# %% [markdown]
# ## Why `exp(t log M)` needs care
#
# `scipy.linalg.logm` returns the **principal** logarithm, whose branch cut runs along the negative
# real axis. A half-turn has both eigenvalues at $-1$, sitting right on that cut — so the principal
# log comes back **complex** ($\log(-I) = i\pi I$) even though the perfectly real rotation generator
# $\begin{pmatrix} 0 & -\pi \\ \pi & 0\end{pmatrix}$ exists and does the job.
#
# `clmmathtools.viz.animate.real_logm` takes the principal log when it is real and special-cases the rest. A
# real logarithm exists **iff** every negative eigenvalue has an *even* number of Jordan blocks —
# which is exactly why $-I$ (two $1\times1$ blocks for the eigenvalue $-1$) works, while
# $\mathrm{diag}(-1,-2)$ (one block each) genuinely has none.
#
# Facts: every square matrix can be built out of Jordan blocks
#
# Example of a 2 × 2 Jordan block:
# $$
# \left(\begin{matrix}\lambda &1\\ 0&\lambda \end{matrix}\right)$$
#
# The logarithm of a Jordan block $\mathbf{J} = \lambda \mathbf{I} + \mathbf{N}$ is found using the series:
# $$
# \log(\mathbf{J}) = \log(\lambda \mathbf{I} + \mathbf{N})$$.
# This expands to:
# $$
#  \log(\lambda)\mathbf{I} + \sum_{k=1}^{\infty} \frac{(-1)^{k-1}}{k} (\frac{\mathbf{N}}{\lambda})^k
#  $$.
#
#  If $λ ≤ 0$, the term $\log(\lambda)$ is not a real number (it is complex or undefined).
#  - If $λ = 0$, the matrix $\mathbf{N}/\lambda$ division is undefined because you cannot divide by zero.
#  - If a Jordan block has a zero or negative eigenvalue, its logarithm requires complex numbers or breaks down completely.
#  - Therefore, the whole matrix cannot have a real-valued logarithm.
#
# ### The Negative Eigenvalue Trap for `scipy.linalg.logm`:
#
# If a real matrix has a negative real eigenvalue (e.g., $λ = -2$), the principal branch maps its logarithm to $\ln(2) + i\pi$. Because of this explicit choice, the final computed matrix will always be complex-valued, even if a non-principal branch exists that would yield a purely real matrix.2. When Do Real Non-Principal Branches Exist?As a reminder, a real matrix has a real logarithm if and only if it is invertible and every Jordan block belonging to a negative real eigenvalue occurs an even number of times.For example, consider the matrix:
# $$
# A=\left[\begin{matrix}-1&0\\ 0&-1\end{matrix}\right]$$
# The principal branch gives a complex result:
# $$\log _{principal}(A)=\left[\begin{matrix}i\pi &0\\ 0&i\pi \end{matrix}\right]$$
# However, because the eigenvalue -1 has a twin, we can assign +iπ to one block and -iπ to the other. Complex conjugate pairs in the matrix eigenvalues allow the resulting matrix to be entirely real:
#
#
# References:
# - [Wikipedia Matrix Logarithm](https://en.wikipedia.org/wiki/Logarithm_of_a_matrix)
# - [MathWorld Reference: Jordan block form](https://www.mathwords.com/j/jordan_block.htm)
# - [Dieci, Morini and Papini. COMPUTATIONAL TECHNIQUES FOR REAL LOGARITHMS OF MATRICES](https://dieci.math.gatech.edu/preps/DMP-Log.pdf)
# - [logm scipy.linalg.logm manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html)
#
#  Conditions for real logarithm:
#  - Invertibility: The matrix cannot be singular (zero is not allowed as an eigenvalue).
#  - Negative Eigenvalues: Negative eigenvalues are allowed, but their corresponding Jordan blocks must pair up evenly. If a negative eigenvalue has an odd number of Jordan blocks, any logarithm of that matrix will contain complex or non-real numbers.
#  - Principal Logarithm: If a real matrix has no negative eigenvalues and no zero eigenvalues, it possesses a unique "principal" logarithm that is completely real.
#

# %%
# print("principal logm(-I) — complex, so exp(t·logm(-I)) collapses just like the lerp:")
mv.show_md(
    r"principal $\operatorname{logm}(-I)$ is complex, so $\exp(t \cdot \operatorname{logm}(-I))$ collapses just like the lerp:"
)
mv.show_expr(sp.Matrix(np.round(logm(HALF_TURN), 3)))
# print("\nreal_logm(-I) — the π-rotation generator:")
mv.show_md("real_logm(-I) — the $\pi$-rotation generator:")
mv.show_expr(sp.Matrix(np.round(real_logm(HALF_TURN), 3)), label="L")
print("\ncheck: expm(L) == -I ?", np.allclose(expm(real_logm(HALF_TURN)), HALF_TURN))

# diag(-1,-2): det > 0, but no real logarithm exists at all
try:
    real_logm(np.diag([-1.0, -2.0]))
except ValueError as e:
    print("\ndiag(-1,-2) →", e)

# %% [markdown]
# ## The geodesic really is a one-parameter subgroup
#
# The defining property is $M(s)\,M(t) = M(s+t)$ — the path is a *homomorphism* from
# $(\mathbb{R}, +)$ into the group. And $\det M(t) = e^{\,t\operatorname{tr} L}$, which is positive
# for every $t$ no matter what $L$ is: **that** is the structural reason the geodesic can never
# degenerate. Verified numerically, in the house style of this repo:

# %%
at, L = geodesic_path(GROW), real_logm(GROW)
print("M(s)M(t) == M(s+t)          ?", np.allclose(at(0.3) @ at(0.4), at(0.7)))
print(
    "det M(t) == exp(t·tr L)     ?",
    np.allclose([np.linalg.det(at(t)) for t in ts], np.exp(ts * np.trace(L))),
)
# for a *pure rotation* the polar split is R·I, so the two principled paths coincide exactly
print(
    "polar == geodesic on -I     ?",
    all(np.allclose(polar_path(HALF_TURN)(t), geodesic_path(HALF_TURN)(t)) for t in ts),
)

# %% [markdown]
# ## The geodesic fixes eigendirections
#
# Because $\exp(t\log M)$ is a polynomial in $M$, it shares $M$'s eigenvectors. So an eigendirection
# of $M$ is invariant at *every* $t$: the geodesic scales it by $\lambda^{t}$ — exponentially and
# monotonically, never overshooting or reversing. Below, the eigendirections are drawn as fixed
# purple lines while the grid flows along them.

# %%
M = np.array([[2.0, 1.0], [0.0, 0.5]])  # eigenvalues 2, 1/2
vals, vecs = np.linalg.eig(M)
at = geodesic_path(M)
print("eigenvalues of M:", np.round(vals, 3))
for k in range(2):
    v, img = vecs[:, k], at(0.5) @ vecs[:, k]
    parallel = (
        abs(img[0] * v[1] - img[1] * v[0]) < 1e-9
    )  # 2-D cross product (np.cross: NumPy<2)
    print(
        f"  M(1/2)·v{k} stays on span(v{k}) ? {parallel}   (scaled by {vals[k] ** 0.5:.3f})"
    )


# %% [markdown]
# ## Any scene, any sweep: `scrubber(build)`
#
# `animate_matrix` is a thin convenience over the general tool: hand `scrubber` *any*
# `build(t) -> Plane | Space3D`.


# %%
def build(t):
    plane = mv.Plane(extent=3, grid=False)
    for k in range(2):  # the invariant eigendirections
        plane.line((0, 0), vecs[:, k], color=mv.PURPLE, width=1.5, alpha=0.5)
    return plane.push(at(t), probe_shape=mv.UNIT_SQUARE, basis=True, samples=2)


mv.scrubber(build, n=24, ping_pong=True)

# %% [markdown]
# ## Sweeping something other than a matrix
#
# $t$ need not drive a linear map at all. Here it morphs a complex function $g_t(z) = z^{\,1 + t}$ —
# the identity at $t = 0$, the squaring map at $t = 1$ — shown as a Wegert phase portrait. Watch the
# single zero at the origin deepen into a double zero as the color wheel wraps twice. (These frames
# are far more expensive, so render fewer of them.)

# %%
mv.scrubber(
    lambda t: mv.Plane(
        extent=1.6, grid=False, axes=False, size=(200, 200)
    ).phase_portrait(lambda z: z ** (1 + t), res=220),
    n=24,
    ping_pong=False,
    interval=140,
)

# %% [markdown]
# ## GIF export
#
# The same builder, written to an animated GIF — portable, renders on GitHub, needs no widget stack.
# This generalizes the hand-rolled shear animation in `Geometry/2dplaneVedo.py`.

# %%
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from IPython.display import Image as IPyImage  # noqa: E402

shear_at = geodesic_path(SHEAR)
out = Path(tempfile.mkdtemp()) / "shear.gif"
mv.to_gif(
    lambda t: mv.Plane(extent=3, grid=False, size=(360, 360)).push(
        shear_at(t), probe_shape=mv.UNIT_SQUARE, basis=True, samples=2
    ),
    out,
    n=20,
    fps=15,
    ping_pong=True,
)
IPyImage(filename=str(out))  # embeds the bytes — no stray file in the repo

# %% [markdown]
# **Recap.**
#
# - `mv.animate_matrix(M, kind=...)` — scrub the linear map $I \to M$.
# - `mv.scrubber(build)` — scrub *any* `build(t) -> Plane | Space3D`; frames pre-rendered offscreen.
# - `mv.to_gif(build, path)` — the same sweep as a portable GIF.
# - `mv.matrix_path(M, kind)` — just the path $t \mapsto M(t)$, if you want to drive your own scene.
#
# And the mathematical point: **`lerp` interpolates the matrix; `geodesic` interpolates the
# transformation.** A straight line in matrix space can walk right through a singular matrix. The
# one-parameter subgroup cannot, because $\det \exp(A) = e^{\operatorname{tr} A} > 0$ always.
