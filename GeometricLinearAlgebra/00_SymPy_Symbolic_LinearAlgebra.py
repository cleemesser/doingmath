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
# # Geometric Linear Algebra 0 — The Symbolic Companion (SymPy)
#
# The notebooks in this series prove their claims numerically, with `numpy` and `np.allclose`. This
# companion re-derives the **same core results symbolically**, with [SymPy](https://www.sympy.org):
# exact fractions instead of `0.333…`, identities proven *as identities* (the difference simplifies to
# a clean $0$), and **symbols placed inside the matrices** so a result holds for *all* inputs at once,
# not just a random sample.
#
# It is a tour of the whole series through one tool:
# 1. **Operators as exact matrices** (nb 2) — the quarter-turn $J$ with $J^2=-I$, rotation, projection,
#    reflection, shear — each property an exact identity.
# 2. **Dot and wedge** (nb 3) — the Pythagorean identity $\langle u,v\rangle^2+(u\wedge v)^2=\rvertu\rvert^2\rvertv\rvert^2$
#    proven symbolically.
# 3. **Determinant and trace** (nb 4) — multiplicativity, $\det$ = area scaling, and the coordinate-free
#    trace $\frac{d}{dt}\big|_0\det(I+tT)$, plus $\det e^{tT}=e^{t\,\operatorname{tr}T}$.
# 4. **Change of basis and ℂ** (nb 5) — similarity preserves $\det$/$\operatorname{tr}$, and the
#    "scale-and-rotate" algebra $xI+yJ$ **is** the complex numbers, shown by exact matrix arithmetic.
#
# A couple of `mathviz` pictures tie the symbols back to geometry.

# %%
import numpy as np
import sympy as sp
from sympy import (
    Matrix,
    symbols,
    eye,
    sqrt,
    cos,
    sin,
    simplify,
    trigsimp,
    factor,
    Rational,
)

sp.init_printing()

import mathviz as mv  # shared plane-viz library (see ../mathviz)

GREY = 0x888888
RED = 0xEF5350
GREEN = 0x33C463
BLUE = 0x4FC3F7
ORANGE = 0xFFB74D
PURPLE = 0xCE93D8
# BG = 0x0F0F0F # green background
BG = 0x000000  # white background?


# Plane drawing now comes from the shared `mathviz` library: mv.Plane(extent=…) is a
# top-down view whose .vector / .curve / .display match what this notebook used.


# %% [markdown]
# ## 1. The operators as exact matrices (notebook 2, symbolically)
#
# The geometric operators were defined coordinate-free; with a basis they become matrices. SymPy lets
# us verify their defining properties as **exact identities**. Start with the quarter-turn $J$ — the
# seed of $i^2=-1$ — and rotation $R(\theta)=\cos\theta\,I+\sin\theta\,J$.

# %%

# %%

# %%

# %%
theta, alpha, beta = symbols("theta alpha beta", real=True)
J = Matrix([[0, -1], [1, 0]])
I2 = eye(2)

print(
    "J =", J.tolist(), "  J² =", (J * J).tolist(), " = −I   (the geometric i² = −1)\n"
)

R = cos(theta) * I2 + sin(theta) * J
mv.show_expr(R)
print("Rᵀ R = I (orthogonal):", trigsimp(R.T * R) == I2)
print("det R = 1            :", trigsimp(R.det()) == 1)
Ra = cos(alpha) * I2 + sin(alpha) * J
Rb = cos(beta) * I2 + sin(beta) * J
Rab = cos(alpha + beta) * I2 + sin(alpha + beta) * J
print("R(α)R(β) = R(α+β)    :", trigsimp(Ra * Rb - Rab) == sp.zeros(2))
print(
    "  ↑ the angle-sum identities for sin and cos, falling out of matrix multiplication"
)

# %% [markdown]
# **Projection, reflection, shear** — each with its defining property as an exact identity. With a
# *symbolic* direction $a=(a_1,a_2)$ the projection's idempotency holds for every line at once.

# %%
a1, a2, k = symbols("a1 a2 k", real=True)
a = Matrix([a1, a2])
P = (a * a.T) / (a.T * a)[0]  # projection onto the line through a
print("projection P_a idempotent (P² = P):", simplify(P * P - P) == sp.zeros(2))
F = 2 * P - I2  # reflection across that line
print("reflection F = 2P − I is an involution (F² = I):", simplify(F * F) == I2)

d = Matrix([1, 0])
n = J * d  # unit normal to the shear line (x-axis)
H = I2 + k * (d * n.T)  # shear of strength k along d
print("shear H fixes its line (H·d = d):", simplify(H * d) == d)
print("shear preserves area (det H = 1):", simplify(H.det()) == 1)

# %% [markdown]
# ## 2. Dot and wedge: the Pythagorean identity, proven (notebook 3)
#
# The dot product (symmetric, $\langle u,v\rangle=|u||v|\cos\theta$) and the wedge (antisymmetric,
# $u\wedge v=|u||v|\sin\theta$, the signed area) are the two halves of multiplying two arrows. Their
# squares add to the product of the squared lengths — the identity
# $\langle u,v\rangle^2+(u\wedge v)^2=\|u\|^2\|v\|^2$ is just $\cos^2+\sin^2=1$ in disguise. SymPy
# proves it for arbitrary $u,v$.

# %%
u1, u2, v1, v2 = symbols("u1 u2 v1 v2", real=True)
u, v = Matrix([u1, u2]), Matrix([v1, v2])
dot = (u.T * v)[0]
wedge = u1 * v2 - u2 * v1  # signed area = 2×2 determinant [u v]
lhs = dot**2 + wedge**2
rhs = (u.T * u)[0] * (v.T * v)[0]
print("⟨u,v⟩² + (u∧v)² − ‖u‖²‖v‖²  =", simplify(lhs - rhs), "  (identically zero ✓)")

# the wedge is exactly the determinant of the matrix [u | v]
print("u ∧ v == det[u v]:", simplify(wedge - u.row_join(v).det()) == 0)

# an exact angle, where numpy would only give a decimal.
# (1,2,2) and (2,2,1) both have norm 3 and dot product 8, so cos = 8/9 exactly — an
# angle with no closed form, held exactly anyway. Note the vectors must NOT be
# orthogonal for this to make its point: (1,2,2)·(2,0,−1) = 2 + 0 − 2 = 0 would give a
# dull π/2.
uu, vv = Matrix([1, 2, 2]), Matrix([2, 2, 1])
cos_ang = (uu.T * vv)[0] / (uu.norm() * vv.norm())
print("\nexact cos(angle) between (1,2,2) and (2,2,1):", cos_ang)
print("exact angle:", sp.acos(cos_ang), "≈", float(sp.acos(cos_ang)), "rad")

# picture: the wedge as the signed area of the parallelogram on u and v
un, vn = np.array([2.0, 0.6]), np.array([0.7, 1.8])
pl = mv.Plane(extent=3)
pl.curve([np.zeros(2), un, un + vn, vn, np.zeros(2)], BLUE, 3)  # parallelogram
pl.vector(un, color=RED, label="u").vector(vn, color=GREEN, label="v")
pl.display()
print(
    f"signed area u∧v = {un[0] * vn[1] - un[1] * vn[0]:.2f}  = the determinant of [u v]"
)

# %% [markdown]
# ## 3. Determinant and trace (notebook 4), exactly
#
# The determinant is **multiplicative** and equals the factor by which signed area scales; the trace
# is the *first-order* area rate, $\operatorname{tr}T=\frac{d}{dt}\big|_0\det(I+tT)$. Both are proven
# below for symbolic matrices, and we close with $\det e^{tT}=e^{t\,\operatorname{tr}T}$.

# %%
a, b, c, dd, e, f, g, h = symbols("a b c d e f g h", real=True)
A = Matrix([[a, b], [c, dd]])
Bm = Matrix([[e, f], [g, h]])
print("det(AB) = det(A)·det(B):", simplify((A * Bm).det() - A.det() * Bm.det()) == 0)

t = symbols("t", real=True)
T = Matrix([[a, b], [c, dd]])
area = (eye(2) + t * T).det()  # signed area of the image of the unit square
print("det(I + tT) =", sp.expand(area))
print("d/dt det(I+tT) at t=0  =", sp.diff(area, t).subs(t, 0), " = a + d = tr(T) ✓")

# det(exp(tT)) = exp(t·tr T), shown on a concrete (triangular) T so the matrix exp is clean
Tt = Matrix([[1, 2], [0, 3]])
expTt = (t * Tt).exp()
print("\nexp(tT) =")
mv.show_expr(expTt)
print(
    "det(exp(tT)) =", simplify(expTt.det()), "   exp(t·tr T) =", sp.exp(t * Tt.trace())
)
print("equal:", simplify(expTt.det() - sp.exp(t * Tt.trace())) == 0)

# parameter inside the matrix: when is it singular?
kk = symbols("k", real=True)
Bk = Matrix([[1, kk], [kk, 4]])
print(
    "\ndet([[1,k],[k,4]]) =", Bk.det(), " → singular when k =", sp.solve(Bk.det(), kk)
)

# %% [markdown]
# ## 4. Change of basis, and discovering ℂ (notebook 5)
#
# Writing an operator in a different basis is the **similarity** $A\mapsto P^{-1}AP$. The determinant
# and trace are **basis-independent** — they are properties of the operator, not its matrix. SymPy
# confirms it for an arbitrary $A$.

# %%
A = Matrix([[a, b], [c, dd]])
P = Matrix([[2, 1], [1, 1]])  # any invertible change of basis
Asim = P.inv() * A * P
print("similar matrix P⁻¹AP =")
mv.show_expr(simplify(Asim))
print("det preserved :", simplify(Asim.det() - A.det()) == 0)
print("trace preserved:", simplify(Asim.trace() - A.trace()) == 0)

# %% [markdown]
# **The capstone: $xI+yJ$ *is* $\mathbb{C}$.** The "scale-and-rotate" operators — a real multiple of
# the identity plus a real multiple of the quarter-turn — are closed and commutative under
# multiplication, with $\det = x^2+y^2$. Mapping $xI+yJ \leftrightarrow x+iy$ turns matrix
# multiplication into **complex multiplication**, exactly.

# %%
x, y, x2, y2, phi = symbols("x y x2 y2 phi", real=True)


def Z(xx, yy):
    return xx * I2 + yy * J


prod = Z(x, y) * Z(x2, y2)
print("(xI+yJ)(x₂I+y₂J) =")
mv.show_expr(prod)
print(
    "equals Z(x·x₂ − y·y₂,  x·y₂ + x₂·y):",
    simplify(prod - Z(x * x2 - y * y2, x * y2 + x2 * y)) == sp.zeros(2),
)
print("  ↑ exactly the rule (x+iy)(x₂+iy₂) = (xx₂−yy₂) + (xy₂+x₂y)i\n")
print(
    "commutative  Z₁Z₂ = Z₂Z₁ :",
    simplify(Z(x, y) * Z(x2, y2) - Z(x2, y2) * Z(x, y)) == sp.zeros(2),
)
print("modulus²     det Z = x²+y²:", simplify(Z(x, y).det() - (x**2 + y**2)) == 0)

# e^{φJ} is the rotation — Euler's formula at the matrix level
print("\nexp(φJ) =")
mv.show_expr(trigsimp((phi * J).exp()))
print(
    "equals R(φ) = cosφ·I + sinφ·J :",
    trigsimp((phi * J).exp() - (cos(phi) * I2 + sin(phi) * J)) == sp.zeros(2),
)
print("  ↑ e^{iφ} = cosφ + i sinφ, the complex exponential, as a matrix identity")

# geometry: e^{φJ} rotates the standard basis
ph = np.pi / 5
Rn = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
pl = mv.Plane(extent=2)
pl.vector([1, 0], color=GREY, alpha=0.5).vector([0, 1], color=GREY, alpha=0.5)
pl.vector(Rn @ [1, 0], color=RED, label="e^{φJ} e₁").vector(
    Rn @ [0, 1], color=GREEN, label="e^{φJ} e₂"
)
pl.display()
print(
    "e^{φJ} with φ = 36°: the basis arrows are rotated — the matrix exponential is a rotation."
)

# %% [markdown]
# ## Summary
#
# Every headline result of the series, re-proven exactly with SymPy:
#
# | result | notebook | how SymPy shows it |
# |--------|----------|--------------------|
# | $J^2=-I$, $R(\theta)=\cos\theta\,I+\sin\theta\,J$ orthogonal, $R(\alpha)R(\beta)=R(\alpha{+}\beta)$ | 2 | `trigsimp(...) == 0` |
# | $P_a$ idempotent, $F_a$ involution, shear preserves area | 2 | symbolic direction $a$, exact identity |
# | $\langle u,v\rangle^2+(u\wedge v)^2=\rvert u\lvert^2 \rvert v \lvert^2$, wedge $=\det[u\,v]$ | 3 | `simplify(lhs-rhs) == 0` |
# | $\det(AB)=\det A\det B$, $\operatorname{tr}T=\frac{d}{dt}\big\rvert_0\det(I+tT)$, $\det e^{tT}=e^{t\operatorname{tr}T}$ | 4 | symbolic $A,B,T$; `diff`, `.exp()` |
# | $\det,\operatorname{tr}$ similarity-invariant; $xI+yJ\cong\mathbb{C}$; $e^{\phi J}=R(\phi)$ | 5 | `P.inv()*A*P`, exact complex product |
#
# The lesson of the symbolic view: **a `simplify(... ) == 0` on a matrix of symbols is a proof for all
# inputs**, where `np.allclose` only ever checks one sample. SymPy's exact arithmetic also keeps the
# geometry honest — angles come out as `acos(8/9)`, eigenvalues as `x ± iy`, determinants as `x²+y²` —
# so the structure stays visible instead of dissolving into floating-point noise. Use it alongside the
# numerical notebooks: `numpy` to *see and compute*, SymPy to *prove and understand*.
