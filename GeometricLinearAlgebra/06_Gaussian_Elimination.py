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
# # Geometric Linear Algebra 6 — Gaussian Elimination, Step by Step
#
# Gaussian elimination is the engine of computational linear algebra: it solves systems, inverts
# matrices, computes determinants and ranks, and exposes the four fundamental subspaces. It is just a
# disciplined sequence of three **reversible** moves on the rows of a matrix. The point of this
# notebook is to watch that sequence unfold **one operation at a time** — using SymPy's
# `Matrix.elementary_row_op` so each step is explicit and exact (fractions, not `0.333…`) — and then
# connect the algebra to the geometry: each equation is a line/plane, and elimination is steering them
# to reveal where they meet.
#
# Tool of the trade: SymPy's three elementary row operations,
#
# | op string | effect | call |
# |-----------|--------|------|
# | `"n<->m"` | swap two rows | `M.elementary_row_op("n<->m", row1=i, row2=j)` |
# | `"n->kn"` | scale a row by $k\neq 0$ | `M.elementary_row_op("n->kn", row=i, k=k)` |
# | `"n->n+km"` | add $k\times$ another row | `M.elementary_row_op("n->n+km", row1=i, row2=j, k=k)` |
#
# Each returns a **new** matrix (it does not mutate), so we reassign as we go.

# %%
import numpy as np
import sympy as sp
from sympy import Matrix, Rational, symbols, eye, linsolve

sp.init_printing()

import mathviz as mv  # shared plane-viz library (see ../mathviz)

GREY = 0x888888
RED = 0xEF5350
GREEN = 0x33C463
BLUE = 0x4FC3F7
ORANGE = 0xFFB74D
PURPLE = 0xCE93D8
BG = 0x0F0F0F


# Plane drawing now comes from the shared `mathviz` library. mv.Plane gives a top-down
# view with grid+axes; this one domain-specific helper draws the line a·x + b·y = c
# (the "row picture" of an equation) via mv.Plane.line(point, direction).
def line_eq(pl, a, b, c, color=BLUE):
    """Draw the line a·x + b·y = c across the plane `pl` (direction ⟂ normal (a, b))."""
    p0 = (0.0, c / b) if abs(b) > 1e-12 else (c / a, 0.0)
    return pl.line(p0, (-b, a), color=color)


def show(M, title=""):
    """Pretty-print a labelled matrix step."""
    if title:
        print(title)
    mv.show_expr(M)
    print()


# %% [markdown]
# ## 1. The three operations, made explicit
#
# Every move below is reversible (that is what keeps the solution set unchanged). Watch one of each
# applied to a small matrix.

# %%
M0 = Matrix([[2, 4, 2], [1, 1, 3], [3, 1, 1]])
show(M0, "start:")
show(
    M0.elementary_row_op("n->kn", row=0, k=Rational(1, 2)), "R0 → ½·R0  (scale a row):"
)
show(M0.elementary_row_op("n<->m", row1=0, row2=2), "R0 ↔ R2  (swap two rows):")
show(
    M0.elementary_row_op("n->n+km", row1=1, row2=0, k=-Rational(1, 2)),
    "R1 → R1 − ½·R0  (eliminate an entry):",
)

# %%
scale0 = Matrix([[1/2, 0, 0], [0,1,0],[0,0,1]])
scale0

# %%
scale0 @ M0

# %%
M0 @ scale0

# %%

# %% [markdown]
# ## 2. Forward elimination to row-echelon form
#
# Solve the system
#
# $$\begin{aligned} 2x + y - z &= 8\\ -3x - y + 2z &= -11\\ -2x + y + 2z &= -3 \end{aligned}$$
#
# by carrying its **augmented matrix** $[A\,|\,b]$ to an upper-triangular (echelon) form. Each step
# clears one entry below a pivot using the `"n->n+km"` operation; nothing else changes.

# %%

# %%
Aug = Matrix([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
show(Aug, "augmented [A | b]:")

# clear column 0 below the pivot (2)
Aug = Aug.elementary_row_op(
    "n->n+km", row1=1, row2=0, k=Rational(3, 2)
)  # R1 → R1 + 3/2 R0
Aug = Aug.elementary_row_op("n->n+km", row1=2, row2=0, k=Rational(1, 1))  # R2 → R2 + R0
show(Aug, "after clearing column 0:")

# clear column 1 below the new pivot (1/2)
piv = Aug[1, 1]
Aug = Aug.elementary_row_op("n->n+km", row1=2, row2=1, k=-Aug[2, 1] / piv)
show(Aug, "after clearing column 1  → row-echelon form (upper triangular):")

print("Back-substitution from the bottom row up:")
z = Aug[2, 3] / Aug[2, 2]
y = (Aug[1, 3] - Aug[1, 2] * z) / Aug[1, 1]
x = (Aug[0, 3] - Aug[0, 1] * y - Aug[0, 2] * z) / Aug[0, 0]
print(f"  z = {z},  y = {y},  x = {x}   → (x, y, z) = ({x}, {y}, {z})")

# %% [markdown]
# ## 3. All the way to reduced row-echelon form (RREF), and a cross-check
#
# Continuing — scale each pivot to $1$ and clear *above* the pivots too — lands in **reduced**
# row-echelon form, where the solution is read straight off the last column. We do it by hand with
# row ops, then confirm against SymPy's one-shot `rref()`.

# %%
R = Matrix([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
# forward sweep
R = R.elementary_row_op("n->n+km", row1=1, row2=0, k=Rational(3, 2))
R = R.elementary_row_op("n->n+km", row1=2, row2=0, k=1)
R = R.elementary_row_op("n->n+km", row1=2, row2=1, k=-R[2, 1] / R[1, 1])
# normalize pivots to 1
R = R.elementary_row_op("n->kn", row=0, k=Rational(1, 2))
R = R.elementary_row_op("n->kn", row=1, k=1 / R[1, 1])
R = R.elementary_row_op("n->kn", row=2, k=1 / R[2, 2])
# back-clear above the pivots
R = R.elementary_row_op("n->n+km", row1=1, row2=2, k=-R[1, 2])
R = R.elementary_row_op("n->n+km", row1=0, row2=2, k=-R[0, 2])
R = R.elementary_row_op("n->n+km", row1=0, row2=1, k=-R[0, 1])
show(R, "hand-computed RREF:")

rref_M, pivots = Matrix([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]).rref()
show(rref_M, "SymPy rref():")
print("hand RREF matches SymPy rref():", R == rref_M, "  pivot columns:", pivots)

# %% [markdown]
# ## 4. The three things that can happen
#
# A linear system has **exactly one**, **no**, or **infinitely many** solutions — and elimination
# tells which by the shape of the echelon form. SymPy's `linsolve` returns the solution set directly,
# handing back a *parametrized* family in the underdetermined case.

# %%
x, y = symbols("x y")

print("(a) UNIQUE — pivots in every column, lines cross at one point:")
print("    ", linsolve([x + 2 * y - 4, 3 * x - y - 5], x, y), "\n")

print("(b) NONE — elimination hits a contradiction row [0 0 | nonzero]:")
incons = Matrix([[1, 1, 2], [1, 1, 5]])  # x+y=2 and x+y=5
show(incons.rref()[0], "    rref of [x+y=2 ; x+y=5]  → bottom row says 0 = 1:")
print("    linsolve:", linsolve([x + y - 2, x + y - 5], x, y), "(empty set)\n")

print("(c) INFINITELY MANY — a free column, solutions form a line:")
dep = Matrix([[1, 1, 2], [2, 2, 4]])  # x+y=2 and 2x+2y=4 (same line)
show(dep.rref()[0], "    rref of [x+y=2 ; 2x+2y=4]  → one pivot, y is free:")
print(
    "    linsolve:",
    linsolve([x + y - 2, 2 * x + 2 * y - 4], x, y),
    "(parametrized by y)",
)

# geometry of the three cases: each equation is a line; the solution set is their intersection
for eqs, title, col in [
    ([(1, 2, 4, BLUE), (3, -1, 5, GREEN)], "(a) unique: lines cross", ORANGE),
    ([(1, 1, 2, BLUE), (1, 1, 5, GREEN)], "(b) none: parallel lines", None),
    ([(1, 1, 2, BLUE), (2, 2, 4, GREEN)], "(c) infinite: same line", None),
]:
    pl = mv.Plane(extent=5)
    for a, b, c, lc in eqs:
        line_eq(pl, a, b, c, color=lc)
    if title.startswith("(a)"):
        pl.points([[2, 1]], ORANGE, size=11).text(
            [2.2, 1.1], "(2,1)", ORANGE
        )  # unique solution
    print(title)
    pl.display()

# %% [markdown]
# ## 5. Row operations *are* matrices — and elimination inverts a matrix
#
# Each elementary row operation equals **left-multiplication by an elementary matrix** $E$ (apply the
# very same op to the identity to get it). So a whole elimination is one product
# $E_k\cdots E_1 A = R$. Run the sweep on $[A\,|\,I]$: when the left block reaches $I$, the right block
# has become $A^{-1}$ — because the accumulated $E_k\cdots E_1$ *is* $A^{-1}$.

# %%
A = Matrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
# one elementary matrix, to make the "op = matrix" point concrete
E = eye(3).elementary_row_op("n->n+km", row1=1, row2=0, k=-Rational(1, 2))
show(E, "elementary matrix for  R1 → R1 − ½R0  (identity with the op applied):")
print(
    "E·A applies that op to A:",
    (E * A) == A.elementary_row_op("n->n+km", row1=1, row2=0, k=-Rational(1, 2)),
    "\n",
)

# invert by reducing [A | I] → [I | A⁻¹]
aug = A.row_join(eye(3))
red, _ = aug.rref()
Ainv = red[:, 3:]
show(Ainv, "right half of rref([A | I])  =  A⁻¹:")
print("A · A⁻¹ = I :", (A * Ainv) == eye(3), "   matches A.inv():", Ainv == A.inv())

# %% [markdown]
# ## 6. The determinant falls out of elimination
#
# If you reach an upper-triangular form using only **row-replacement** moves (`"n->n+km"`, which do
# not change the determinant) and **row swaps** (each flips its sign), then
#
# $$\det A = (-1)^{\#\text{swaps}} \times (\text{product of the pivots}).$$

# %%
B = Matrix([[0, 2, 1], [1, 1, 1], [2, 1, 0]])  # leading 0 forces a swap
U = B.copy()
swaps = 0
U = U.elementary_row_op("n<->m", row1=0, row2=1)  # swap to get a nonzero pivot
swaps += 1
U = U.elementary_row_op("n->n+km", row1=2, row2=0, k=-U[2, 0] / U[0, 0])
U = U.elementary_row_op("n->n+km", row1=2, row2=1, k=-U[2, 1] / U[1, 1])
show(U, "upper-triangular form (after 1 swap):")
pivot_product = U[0, 0] * U[1, 1] * U[2, 2]
print(
    f"(-1)^{swaps} × (pivot product {pivot_product}) = {(-1) ** swaps * pivot_product}"
)
print(
    "SymPy B.det() =", B.det(), "  → match:", (-1) ** swaps * pivot_product == B.det()
)

# %% [markdown]
# ## Summary
#
# Gaussian elimination is three reversible row moves, applied with a plan:
#
# - **Forward sweep** (`"n->n+km"`) clears below each pivot → row-echelon (upper-triangular) form;
#   **back-substitute** for the answer. Normalizing pivots and clearing *above* them gives **RREF**,
#   where the solution is read off directly (`rref()` does it in one shot).
# - The echelon shape classifies the system: a pivot in every variable column ⇒ **unique**; a
#   $[0\cdots0\,|\,\text{nonzero}]$ row ⇒ **no solution**; a **free** (pivot-less) column ⇒ **infinitely
#   many**, which `linsolve` returns parametrized. Geometrically: lines/planes crossing at a point,
#   never meeting, or coinciding.
# - Each operation is left-multiplication by an **elementary matrix**; chaining them reduces $[A\,|\,I]$
#   to $[I\,|\,A^{-1}]$, so elimination **inverts** a matrix — and the **determinant** is the signed
#   product of the pivots. The same engine, read three different ways.
#
# This is the computational complement to the coordinate-free chapters: once a basis is fixed
# (notebook 5), elimination is how every concrete question — solve, invert, rank, $\det$ — is actually
# answered, with SymPy keeping every intermediate quantity exact.
