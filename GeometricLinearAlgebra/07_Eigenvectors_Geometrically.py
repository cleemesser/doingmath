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
# # Geometric Linear Algebra 7 — Eigenvectors & Eigenvalues, Geometrically
#
# Most arrows, when hit by a linear map, get **turned**: the output points in a different direction
# than the input. But for almost every map there are a few special directions that are *not* turned —
# the map only **stretches** them along their own line. Those directions are the **eigenvectors**, and
# the stretch factor is the **eigenvalue**:
#
# $$A\,v = \lambda\,v \qquad (v \neq 0).$$
#
# This notebook builds that picture geometrically. We *watch* the unit circle of input directions
# become an ellipse and pick out the arrows that stay on their own line; then we use **SymPy** to get
# the eigenvalues and eigenvectors **exactly** (clean integers, not decimals), see the characteristic
# polynomial $\det(A-\lambda I)$ factor, and meet the three regimes a real $2\times2$ map can fall
# into — two real axes, a **defective** shear, and a **rotation** whose eigenvalues are complex
# (closing the loop on the $J^2=-I$ story from the earlier notebooks).
#
# Pictures use the shared **`mathviz`** plane library (static renders in the $z=0$ plane);
# every algebraic claim is cross-checked with SymPy's exact arithmetic.

# %%
import numpy as np
import sympy as sp
from sympy import Matrix, Rational, symbols, I, simplify

sp.init_printing()

import mathviz as mv  # shared plane-viz library (see ../mathviz)

GREY = 0x888888
RED = 0xEF5350
GREEN = 0x33C463
BLUE = 0x4FC3F7
ORANGE = 0xFFB74D
PURPLE = 0xCE93D8
YELLOW = 0xFFD54F
BG = 0x0F0F0F


# Plane drawing now comes from the shared `mathviz` library: mv.Plane(extent=…) gives a
# top-down view with grid+axes; .vector/.curve/.segment/.line/.display match what this
# notebook used (the former inline Plane2D.eigenline(d) is mv.Plane.line([0, 0], d)).


UNIT_CIRCLE = np.c_[
    np.cos(np.linspace(0, 2 * np.pi, 120)), np.sin(np.linspace(0, 2 * np.pi, 120))
]

# %% [markdown]
# ## 1. The question: which arrows keep their direction?
#
# Apply a map $A$ to a fan of unit arrows pointing in every direction. Generically each output arrow
# is rotated away from its input. We draw the input arrows faint and their images bold: the
# **eigen-directions** are exactly the spots where the bold arrow lies *along the same line* as the
# faint one (possibly longer, shorter, or flipped, but not turned off the line).

# %%
A = np.array([[2.0, 1.0], [0.0, 3.0]])  # our running example (eigenvalues 2 and 3)

pl = mv.Plane(extent=4)
for ang in np.linspace(0, np.pi, 12, endpoint=False):  # a fan of input directions
    u = np.array([np.cos(ang), np.sin(ang)])
    pl.vector(u, color=GREY, alpha=0.5)  # input direction (faint)
    pl.vector(A @ u, color=BLUE, alpha=0.9)  # its image (bold)
# the two invariant directions of A, highlighted
pl.line([0, 0], [1, 0], PURPLE).line([0, 0], [1, 1], ORANGE)
pl.display()
print(
    "Most arrows turn (grey → blue). Two lines are special: along them the image stays on the line."
)
print("Those purple/orange lines are the eigen-directions of A = [[2,1],[0,3]].")

# %% [markdown]
# A cleaner way to see it: the unit **circle** of inputs becomes an **ellipse**. Along an eigenvector
# the ellipse touches a ray straight out from the origin (no sideways swing), and the eigenvalue is
# how far out it reaches relative to the input.

# %%
pl = mv.Plane(extent=4)
pl.curve(UNIT_CIRCLE, GREY, 2, 0.6)  # unit circle of inputs
pl.curve(UNIT_CIRCLE @ A.T, BLUE, 3)  # image ellipse
for d, c, lab in [
    (np.array([1.0, 0]), PURPLE, "λ=2"),
    (np.array([1, 1]) / np.sqrt(2), ORANGE, "λ=3"),
]:
    pl.line([0, 0], d, c)
    pl.vector(d, color=c)  # the eigenvector (input)
    pl.vector(
        A @ d, color=c, label=lab, alpha=0.6
    )  # its stretched image, still on the line
pl.display()
print(
    "Circle → ellipse. Eigenvectors are where input and image share a line; λ = stretch factor."
)

# %% [markdown]
# ## 2. The eigenvalue equation and the characteristic polynomial
#
# $A v = \lambda v$ means $(A - \lambda I)v = 0$ with $v\neq 0$, so $A-\lambda I$ must be **singular**:
#
# $$\det(A - \lambda I) = 0.$$
#
# This is the **characteristic polynomial**. SymPy gives it — and the eigenvalues/eigenvectors —
# *exactly*, which is the whole pedagogical point: clean integers instead of `0.7071…`.

# %%
As = Matrix([[2, 1], [0, 3]])
lam = symbols("lambda")
charpoly = (As - lam * sp.eye(2)).det()
print("A =")
sp.pprint(As)
print("\ncharacteristic polynomial det(A − λI) =", sp.factor(charpoly), "= 0")
print("eigenvalues with multiplicity:", As.eigenvals())
print("\neigenvects() — (eigenvalue, multiplicity, [eigenvectors]):")
for val, mult, vecs in As.eigenvects():
    print(f"  λ = {val}  (mult {mult}) → eigenvector {vecs[0].T.tolist()[0]}")

# verify the definition A v = λ v exactly, and read off det = ∏λ, tr = ∑λ
for val, mult, vecs in As.eigenvects():
    v = vecs[0]
    print(
        f"  check  A·v − λ·v = {list((As * v - val * v).T)}  (zero ⇒ genuine eigenvector)"
    )
print("\ndet A =", As.det(), "= 2·3 = product of eigenvalues")
print("tr  A =", As.trace(), "= 2+3 = sum of eigenvalues")

# %% [markdown]
# ## 3. Diagonalization: eigen-coordinates make the map a pure scaling
#
# Stack the eigenvectors as the columns of $P$. In *that* basis the map is just independent stretches —
# the diagonal matrix $D$ of eigenvalues — so
#
# $$A = P\,D\,P^{-1}.$$
#
# Geometrically: change to eigen-coordinates, scale each axis by its eigenvalue, change back. SymPy
# builds $P$ and $D$ and we confirm the factorization exactly.

# %%
P, D = As.diagonalize()
print("P (eigenvectors as columns) =")
sp.pprint(P)
print("D (eigenvalues on the diagonal) =")
sp.pprint(D)
print("P D P⁻¹ == A :", (P * D * P.inv() == As))
print("\nConsequence — powers are easy:  Aⁿ = P Dⁿ P⁻¹")
n = 5
print(f"A^{n} =")
sp.pprint(As**n)
print(f"P D^{n} P⁻¹ =")
sp.pprint(simplify(P * D**n * P.inv()))

# %% [markdown]
# ## 4. Three regimes a real 2×2 map can fall into
#
# ### (a) Symmetric ⇒ perpendicular eigen-axes (the spectral theorem)
#
# A **symmetric** matrix always has real eigenvalues and **orthogonal** eigenvectors. Those are the
# **principal axes** of the ellipse — the directions of maximum and minimum stretch.

# %%
S = Matrix([[2, 1], [1, 2]])
print("Symmetric S =")
sp.pprint(S)
print("eigen-data:")
vecs_S = []
for val, mult, vecs in S.eigenvects():
    print(f"  λ = {val} → eigenvector {vecs[0].T.tolist()[0]}")
    vecs_S.append((float(val), np.array(vecs[0].T.tolist()[0], dtype=float)))
v1, v2 = vecs_S[0][1], vecs_S[1][1]
print("eigenvectors orthogonal (v₁·v₂ = 0):", int(np.dot(v1, v2)) == 0)

Sn = np.array(S.tolist(), dtype=float)
pl = mv.Plane(extent=4)
pl.curve(UNIT_CIRCLE, GREY, 2, 0.6).curve(UNIT_CIRCLE @ Sn.T, BLUE, 3)
for (lamv, d), c, lab in zip(
    vecs_S, (PURPLE, ORANGE), ("λ=3 (long axis)", "λ=1 (short axis)")
):
    dn = d / np.linalg.norm(d)
    pl.line([0, 0], dn, c)
    pl.vector(lamv * dn, color=c, label=lab)
pl.display()
print(
    "Symmetric ⇒ the eigen-axes are perpendicular = the ellipse's principal axes (PCA lives here)."
)

# %% [markdown]
# ### (b) Shear ⇒ **defective**: a repeated eigenvalue but only one eigenvector
#
# A shear $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ fixes the $x$-axis and slides everything parallel to
# it. Its only invariant direction is that axis: $\lambda = 1$ is a **double** root of the
# characteristic polynomial, yet it has just **one** eigenvector. Such a matrix is *defective* — it
# **cannot** be diagonalized.

# %%
H = Matrix([[1, 1], [0, 1]])
print("Shear H =")
sp.pprint(H)
print(
    "char. poly:",
    sp.factor((H - lam * sp.eye(2)).det()),
    " → λ = 1 (algebraic multiplicity 2)",
)
print(
    "eigenvects:",
    [(val, len(vecs)) for val, mult, vecs in H.eigenvects()],
    "→ only ONE eigenvector (1,0)",
)
try:
    H.diagonalize()
    print("diagonalizable")
except Exception as e:
    print(
        "H.diagonalize() raises:", type(e).__name__, "— defective, not diagonalizable"
    )

Hn = np.array(H.tolist(), dtype=float)
pl = mv.Plane(extent=4)
pl.curve(UNIT_CIRCLE, GREY, 2, 0.6).curve(UNIT_CIRCLE @ Hn.T, BLUE, 3)
pl.line([0, 0], [1, 0], PURPLE)
pl.vector([1, 0], color=PURPLE, label="only eigenvector")
for ang in [
    np.pi / 3,
    np.pi / 2,
    2 * np.pi / 3,
]:  # other directions all get sheared off their line
    u = np.array([np.cos(ang), np.sin(ang)])
    pl.vector(u, color=GREY, alpha=0.5).vector(Hn @ u, color=BLUE, alpha=0.9)
pl.display()
print(
    "Only the x-axis survives as an invariant line; every other arrow is slid off its own direction."
)

# %% [markdown]
# ### (c) Rotation ⇒ **complex** eigenvalues: no real eigenvector at all
#
# A pure rotation turns *every* arrow, so it has **no** real eigenvector. Its characteristic
# polynomial has no real roots — the eigenvalues are the complex pair $e^{\pm i\theta}$. This is the
# same $J^2 = -I$ that seeded the complex numbers in notebooks 2 & 5: the quarter-turn's eigenvalues
# are $\pm i$, and a "scale-and-rotate" $xI + yJ$ has eigenvalues $x \pm iy$ — *the complex number
# itself*.

# %%
theta = symbols("theta", real=True)
Rsym = Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
print(
    "Rotation R(θ) eigenvalues:", simplify(Rsym.eigenvals())
)  # e^{±iθ} as cosθ ± i sinθ

J = Matrix([[0, -1], [1, 0]])  # the quarter-turn
print("\nQuarter-turn J = [[0,-1],[1,0]],  J² =", (J * J).tolist(), " (= −I)")
print("eigenvalues of J:", J.eigenvals(), " → ± i  (purely imaginary)")

x, y = symbols("x y", real=True)
Z = x * sp.eye(2) + y * J  # the 'scale-and-rotate' operator xI + yJ
print(
    "\nScale-and-rotate  xI + yJ  has eigenvalues:",
    Z.eigenvals(),
    " = x ± iy  (the complex number itself)",
)
print("|z|² = det(xI+yJ) =", Z.det(), "= x² + y²   (modulus squared)")

# geometric view: a rotation sends the circle to itself but swings every arrow — no fixed line
Rn = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
pl = mv.Plane(extent=3)
pl.curve(UNIT_CIRCLE, GREY, 2, 0.6).curve(UNIT_CIRCLE @ Rn.T, BLUE, 3)
for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    u = np.array([np.cos(ang), np.sin(ang)])
    pl.vector(u, color=GREY, alpha=0.45).vector(Rn @ u, color=BLUE, alpha=0.85)
pl.display()
print(
    "Rotation by 40°: the image circle coincides with the input, but EVERY arrow is turned —"
)
print("no arrow stays on its line ⇒ no real eigenvector ⇒ complex eigenvalues e^{±iθ}.")

# %% [markdown]
# ## Summary
#
# An **eigenvector** is a direction a linear map does not turn; the **eigenvalue** is the factor it
# stretches that direction by. Everything else in the chapter is a consequence:
#
# | map | eigenvalues | eigenvectors | picture |
# |-----|-------------|--------------|---------|
# | symmetric $\begin{smallmatrix}2&1\\1&2\end{smallmatrix}$ | $3,\ 1$ (real) | **orthogonal** | principal axes of the ellipse |
# | shear $\begin{smallmatrix}1&1\\0&1\end{smallmatrix}$ | $1$ (double) | **one** (defective) | only the $x$-axis is invariant |
# | rotation $R(\theta)$ | $e^{\pm i\theta}$ (complex) | **none real** | every arrow turns |
#
# - **Find them exactly:** $\det(A-\lambda I)=0$ for the eigenvalues, then $(A-\lambda I)v=0$ for the
#   eigenvectors — SymPy's `charpoly`, `eigenvals`, `eigenvects` do this with exact arithmetic.
# - **Diagonalize:** when there are enough independent eigenvectors, $A = PDP^{-1}$ turns the map into
#   independent scalings — making $A^n$, matrix exponentials, and stability trivial.
# - **$\det = \prod\lambda$, $\operatorname{tr} = \sum\lambda$** — the volume scaling (notebook 4) and
#   the trace are just the product and sum of the eigenvalues.
# - **The complex thread returns:** a map with no invariant real direction has complex eigenvalues; the
#   rotation's are $e^{\pm i\theta}$, and the scale-and-rotate $xI+yJ$ has eigenvalues $x\pm iy$ — the
#   complex numbers, recovered one last time from pure geometry.
