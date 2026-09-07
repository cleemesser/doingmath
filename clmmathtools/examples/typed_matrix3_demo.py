# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: doingmath (3.14.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Typed matrices, take 3: splitting the space from the frame
#
# This is a port of `typed_matrix2_demo` onto `clmmathtools.typed_matrix3`, which
# replaces the single `Basis` type with two:
#
# | | `typed_matrix` | `typed_matrix3` |
# |---|---|---|
# | the space | implicit in `Basis` | `VectorSpace` --- dimension and $g$ |
# | one basis of it | `Basis` | `Frame` --- its vectors, in reference components |
# | a frame's Gram matrix | *declared* per basis | **derived**: $g_F = B^T g_{\mathrm{ref}} B$ |
# | change of frame | supplied, then checked | **computed**: $P = B_t^{-1}B_s$ |
#
# The motivation is the metric. In `typed_matrix` each `Basis` carried its own
# `metric=`, so two bases of what you *meant* to be one space were two
# independent declarations of geometry, reconciled only if you happened to route
# them through `change_of_basis(check=True)`. A metric is a property of the
# space; its matrix is a property of the frame. Here the space states $g$ once
# and every frame's Gram matrix follows, so two frames cannot disagree --- only
# one of them is doing the talking.
#
# The rest of the notation is unchanged: a matrix prints with its row frame on
# the opening bracket and its column frame on the closing one, raised for
# contravariant and lowered for covariant, so $A^i{}_j$, $g_{ij}$ and $v^i$ read
# off the page.
#
# **Notation in this notebook** follows Misner, Thorne and Wheeler: a vector
# carries an arrow, $\vec{b}_1$, and a covector a tilde, $\tilde{b}^{1}$. The
# grid frame keeps the usual hat, $\hat{e}_1$, which does double duty here ---
# an arrow-family accent, and a reminder that those two are genuinely of unit
# length while $\vec{b}_2$ is not.
#
# The point of giving the dual frame its own
# accent is to make clear when we are talking about a vector or covector. Both are valid vectors in $V$ or $V^\ast$
# but relative to the base vector space, $V$, a covector is not an arrow, instead it is a sort of measurement on vectors.
# Drawing it as one is the very conflation the up/down bookkeeping exists to prevent.


# %%
import sympy as sp
from clmmathtools.viz import show_expr, show_md

sp.init_printing()
from clmmathtools.typed_matrix3 import (
    VectorSpace,
    Frame,
    TMatrix,
    change_of_basis,
    FrameMismatch,
)

# %% [markdown]
# ## Spaces first, then frames
# Note what each line is responsible for: `VectorSpace` says what geometry
# exists, `frame` says how we are going to write it down.

# %%
City = VectorSpace.euclidean("City", 2)  # the plane the city sits in, g = I
E = City.frame("E")  # the grid frame: one block east, one block north

Bare = VectorSpace("Bare", 2)  # a space with NO metric declared
f = Bare.frame("f")

SpaceTime = VectorSpace.pseudo_euclidean("M", [1, -1])  # g = diag(1, -1)
M = SpaceTime.frame("M")

City, E, Bare, f, SpaceTime, M

# %%
# is_orthonormal and self_dual are now relations between a frame and its space's
# metric, computed on demand -- not facts a basis asserts about itself.
for fr in (E, f, M):
    print(
        f"{fr.name}: space={fr.space!r:22} orthonormal={fr.is_orthonormal!s:5} "
        f"self_dual={fr.self_dual}"
    )

# %%
A = TMatrix.operator(sp.Matrix([[2, 1], [0, 3]]), E)
u = TMatrix.vector([1, 2], E)
w = TMatrix.vector([3, 4], E)
A, u, w

# %%
print(sp.latex(A))
print(sp.latex(u))
u.T * w  # two 'up' slots: legal here only because E is self-dual

# %%
# the same contraction in a space with no metric -> rejected
uf = TMatrix.vector([1, 2], f)
try:
    uf.T * TMatrix.vector([3, 4], f)
except FrameMismatch as err:
    print("no metric ->", err)

# %%
# orthonormal but indefinite: variance still matters
uM = TMatrix.vector([1, 2], M)
try:
    uM.T * TMatrix.vector([3, 4], M)
except FrameMismatch as err:
    print("Minkowski ->", err)

print("correct Minkowski pairing:", (uM.T.lower() * TMatrix.vector([3, 4], M)))
uM.T.lower() * TMatrix.vector([3, 4], M)

# %% [markdown]
# ### A simple non-orthogonal frame example: a city with a grid *and* a Broadway
#
# #### The City of Gridville
# ![Map of a hypothetical grid city: numbered streets run north-south, lettered streets run east-west, and a wide Broadway cuts diagonally across the grid](figures/city_broadway_map.svg)
#
# Addresses in this city are quoted in a frame. Put the origin at Founders
# Square (1st & A) and measure in blocks: $\hat{e}_1$ is one block east,
# $\hat{e}_2$ is one block north. "7th & C" is then the pair of numbers
# $\left[\begin{matrix}6\\2\end{matrix}\right]_{\mathcal{E}}$ relative to the
# grid frame $\mathcal{E}$ --- six blocks east and two north of Founders
# Square. (Street *names* are 1-based labels while coordinates are
# displacements, hence 7th $\to 6$ and C $\to 2$.)
#
# But a cab driver on Broadway naturally counts in *Broadway blocks*, a frame
# $\mathcal{B}$ whose second vector points along the diagonal --- and those two
# vectors are neither orthogonal to each other nor both of unit length. The same
# corner then carries two different columns of numbers, and only the subscript
# tells you which city you are counting in.
#
# (The figure is generated by `figures/make_city_map.py`.)

# %% [markdown]
# ### The three accents
# From here on the notebook is full of named vectors and covectors, so the
# Misner-Thorne-Wheeler convention from the top is worth restating as a legend:
#
# | symbol | is | in Gridville |
# |---|---|---|
# | $\hat{e}_i$ | a *unit* vector | one block east, one block north |
# | $\vec{b}_i$ | a vector | one Broadway block, of length $\sqrt{2}$ |
# | $\tilde{b}^{i}$ | a covector | "how many Broadway blocks from here?" |
#
# MTW picture a tensor as a **machine with slots**: drop a vector into each slot
# and a number falls out. A covector is the one-slot machine --- feed it
# $\vec{b}_2$, get a number back. That is the same word `lower(slot=0)` uses
# below, and not by coincidence: a slot of a `TMatrix` is an input the machine
# is still waiting for, which is why a fully contracted result has none left ---
# at which point it stops being a `TMatrix` at all and is simply a number.

# %% [markdown]
# Broadway runs one block east for every block north, so a *Broadway block* is
# just the diagonal of a single city block, and the avenue meets the grid
# squarely at every intersection along it: Founders Square (1st & A), Central
# Square (5th & E), Uptown Square (9th & I). Keeping the numbered-street
# direction as the first vector, the Broadway frame is
# $$\vec{b}_1 = \hat{e}_1 \quad \text{(one block east)}, \qquad
#   \vec{b}_2 = \hat{e}_1 + \hat{e}_2 \quad \text{(one Broadway block)}.$$
# This is the frame worked by hand in
# `GeometricLinearAlgebra/change-basis-no-numpy.ipynb`, so the numbers below
# should match the `B` and `Gb` there.

# %%
# A frame of City -- its columns are b1, b2 in E-components. It declares no
# geometry of its own; there is only one metric here and City owns it.
Bway = City.frame("B", [[1, 1],
                        [0, 1]])  # fmt: skip
#                        ^  ^-- b2 = one Broadway block (1 east, 1 north)
#                        ^-------- b1 = one block east
Bway

# %% [markdown]
# ### The Gram matrix is derived, not declared
# In `typed_matrix2_demo` this had to be written as `gB = Bmat.T * Bmat` and then
# handed to the `Basis` constructor, with a comment insisting it "is not a free
# choice." Now it is not a choice at all --- `Frame.metric` computes
# $g_{\mathcal{B}} = B^T g_{\text{City}} B$ on demand, so there is nowhere to put
# a wrong answer.
#
# Read the entries geometrically: $g_{11} = 1$ (a block east has length one),
# $g_{22} = 2$ (a Broadway block is $\sqrt{1^2 + 1^2} = \sqrt{2}$ blocks long),
# and $g_{12} = 1$, nonzero precisely because Broadway is not perpendicular to
# the numbered streets. So $\mathcal{B}$ is neither orthonormal nor self-dual,
# and every contraction from here on has to go through $g$.

# %%
print("derived Gram matrix:", Bway.metric.tolist())
print("orthonormal:", Bway.is_orthonormal, "| self_dual:", Bway.self_dual)
TMatrix.metric_of(Bway)

# %%
# change_of_basis takes no matrix now: P = B_target^-1 B_source, computed from
# the two frames. There is no isometry check because there is nothing to check.
P = change_of_basis(source=Bway, target=E)
P

# %% [markdown]
# ### Addresses in two frames
# `.in_frame()` is the new front door: it reads the transformation law off each
# slot's variance one slot at a time (row `up` $\to P$, row `down` $\to P^{-T}$,
# col `up` $\to P^{T}$, col `down` $\to P^{-1}$) and leaves alone any slot that
# is already there --- or that belongs to another space entirely.


# %%
def corner(street: int, letter: str) -> TMatrix:
    """Grid-frame position of a corner, in blocks from Founders Square (1st & A)."""
    return TMatrix.vector([street - 1, ord(letter.upper()) - ord("A")], E)


uptown = corner(9, "I")  # Uptown Square, straight up Broadway
home = corner(7, "C")  # off Broadway, southeast of Central Square

uptown_B = uptown.in_frame(Bway)
home_B = home.in_frame(Bway)
assert uptown_B.mat == sp.Matrix([0, 8])  # no east blocks, eight Broadway blocks
assert home_B.mat == sp.Matrix([4, 2])  # four east blocks, two Broadway blocks
assert home_B.in_frame(E).mat == home.mat  # and back again
assert home.to(P.inv()).mat == home_B.mat  # the explicit-P route agrees
uptown_B, home_B

# %% [markdown]
# ### One metric, two matrices
# $g$ is a single coordinate-free object; $g_{ij}$ are its components. Because
# both of its slots are `down`, `.in_frame()` applies a *congruence*
# $P^{-T} g P^{-1}$ and recovers the identity in the grid frame --- nothing had
# to be told that this one is a metric rather than an operator.

# %%
gB = TMatrix.metric_of(Bway)
assert gB.in_frame(E).mat == sp.eye(2)
print("g in B:", gB.mat.tolist(), " ->  g in E:", gB.in_frame(E).mat.tolist())

# and P itself, re-expressed wholly in E, is just the identity operator
assert P.in_frame(E).mat == sp.eye(2)
gB.in_frame(E)

# %% [markdown]
# ### Distances: the naive sum of squares is now wrong
# 7th & C is $\left[\begin{matrix}4\\2\end{matrix}\right]_{\mathcal{B}}$, but it
# is *not* $\sqrt{4^2 + 2^2}$ blocks from Founders Square. `TMatrix` will not
# even let you ask, because two 'up' slots do not contract.

# %%
try:
    home_B.T * home_B
except FrameMismatch as err:
    print("two up-slots ->", err)

print("naive sum of squares (wrong):", (home_B.mat.T * home_B.mat)[0, 0])
print("through the metric:", (home_B.T.lower() * home_B))
print("same thing in the grid frame:", (home.T * home))
assert (home_B.T.lower() * home_B) == (home.T * home)
show_expr(home_B.T.lower() * home_B, label=r"\text{home\_B.T.lower() * home\_B}")

# %% [markdown]
# ### Getting a number back out
# A contraction that uses up every slot *is* a number, and comes back as one ---
# a plain sympy `Integer` here, not a `TMatrix`. That is worth doing because the
# obvious alternative is a $1\times1$ sympy `Matrix`, which is a poor stand-in
# for a number: `M + 1` and `float(M)` raise, and `M == 5` returns `False`
# rather than erroring, so a test written against one silently always fails.
#
# `.scalar` is still there as the explicit route for a zero-slot `TMatrix` built
# by hand, and its guard is the useful half: a leftover slot means the answer is
# a vector or an operator, not a number.

# %%
pairing = home_B.T.lower() * home_B
print("type:", type(pairing).__name__, "| pairing == 40 ->", pairing == 40)
print(
    "the 1x1 matrix it would have been:", sp.Matrix([[40]]) == 40, " (silently False)"
)
try:
    home_B.T.lower().scalar  # a covector, not a number
except FrameMismatch as err:
    print("guard ->", err)

# %% [markdown]
# ### Three ways a contraction can fail
# Splitting space from frame makes the diagnostics much sharper. The old code
# had one message for all of these; now "wrong variance", "right space but wrong
# frame" and "unrelated spaces" are distinguishable, and the middle one can
# suggest the fix.

# %%
Park = VectorSpace.euclidean("Park", 2)  # a different 2-D space entirely
for label, lhs, rhs in (
    ("same frame, same variance", home_B.T, home_B),
    ("same space, different frame", TMatrix.vector([1, 0], E).T, home_B),
    ("different spaces", TMatrix.vector([1, 0], Park.frame("p")).T, home_B),
):
    try:
        lhs * rhs
    except FrameMismatch as err:
        print(f"[{label}]\n    {err}\n")

# %% [markdown]
# ### The dual frame: address readers, not arrows
# $\tilde{b}^{2}$ is the covector answering "how many Broadway blocks from
# Founders Square?" --- it returns $1$ on $\vec{b}_2$ and $0$ on $\vec{b}_1$.
# Pairing a covector with a vector needs no metric at all: the indices already
# match, so the contraction is a plain sum.

# %%
b1_dual = TMatrix.covector([1, 0], Bway)
b2_dual = TMatrix.covector([0, 1], Bway)
print(
    "east blocks      b^1(home) =",
    (b1_dual * home_B),
    "= number E-W blocks from home",
)
print(
    "Broadway blocks  b^2(home) =",
    (b2_dual * home_B),
    "= number broadway blocks from home",
)
show_expr(b2_dual, label=r"\operatorname{b2\_dual}")
show_expr(b2_dual.in_frame(E), r"\rm b2\_dual \, in \, E")

# %% [markdown]
# ### A covector is a stack of streets
# MTW's other picture of a one-form: not an arrow but a **family of parallel
# surfaces**, the number it returns being how many of them the vector pierces.
# In Gridville those surfaces are literally streets.
#
# $\tilde{b}^{2}$, the Broadway-block counter, reads the north coordinate, so its
# level sets are the **lettered streets** --- count how many you cross.
# $\tilde{b}^{1}$, the east-block counter, reads $x - y$, so its level sets are
# the **diagonals running parallel to Broadway**.
#
# Notice what that says: each covector's level sets run parallel to the *other*
# frame vector. That is the entire geometric content of
# $\tilde{b}^{i}(\vec{b}_j) = \delta^i_j$ --- and unlike the word "perpendicular"
# in the next cell, it needs no metric at all. It stays true on a space with no
# inner product, where perpendicularity is not even defined.

# %%
# Checked at a *generic* corner, so this is a proof rather than a sample.
x, y = sp.symbols("x y", real=True)
anywhere = TMatrix.vector([x, y], E).in_frame(Bway)
print("a generic corner (x east, y north), in B:", anywhere.mat.T.tolist())
print(
    "b^1 reads:",
    (b1_dual * anywhere),
    " - constant along Broadway-parallel diagonals",
)
print("b^2 reads:", (b2_dual * anywhere), " - constant along the lettered streets")
assert (b1_dual * anywhere) == x - y
assert (b2_dual * anywhere) == y

# %% [markdown]
# Raising the index turns each reader back into an arrow, and the answers are
# worth staring at. $\tilde{b}^{2\,\sharp} = \hat{e}_2$ is *exactly one block
# north*:
# one Broadway block carries you exactly one block north, so counting north
# blocks already counts Broadway blocks. Note it is perpendicular to
# $\vec{b}_1$, not parallel to $\vec{b}_2$. Likewise
# $\tilde{b}^{1\,\sharp} = \hat{e}_1 - \hat{e}_2$ points southeast, perpendicular to
# Broadway. Each *raised* dual vector is orthogonal to the other frame vector,
# which is the metric restatement of the level sets above, so the dual frame
# leans against the frame --- which is exactly what the up/down
# bookkeeping is tracking.

# %%
for name, dual in (("b^1", b1_dual), ("b^2", b2_dual)):
    arrow = dual.raise_().T  # covector -> vector, then stand it up as a column
    print(f"{name}: {arrow.mat.T} in B   =   {arrow.in_frame(E).mat.T} in E")
b2_dual.raise_().T.in_frame(E)

# %% [markdown]
# ### One map, two matrices
# Orthogonal projection onto Broadway is a single geometric operation. In the
# grid frame it is $\vec{b}_2\vec{b}_2^{\,T} / \lVert \vec{b}_2 \rVert^2$; in the
# Broadway frame it reads
# $\left[\begin{matrix}0 & 0\\ 1/2 & 1\end{matrix}\right]$ --- it keeps the
# Broadway coordinate and picks up half the east coordinate, because a step east
# is not perpendicular to Broadway. Note that it is *not symmetric* --- and yet
# it is still self-adjoint. Symmetry of the bare transpose is a property of the
# numbers in a self-dual frame; self-adjointness is a property of the map.
#
# The projection sends 7th & C to $[0,4]_{\mathcal{B}}$, four Broadway blocks
# from Founders Square: Central Square, 5th & E.

# %%
Proj = TMatrix.operator(sp.Rational(1, 2) * sp.Matrix([[1, 1], [1, 1]]), E)
Proj_B = Proj.in_frame(Bway)
Proj_B

# %% [markdown]
# ### Which slot?
# `lower()` and `raise_()` take a **slot index**, and slot indices count the
# object's *real* slots --- trivial length-1 axes are skipped. A covector stored
# $1\\times n$ therefore has exactly one slot, `slot=0`, even though it occupies
# array position 1: the index describes the tensor, not its layout. That is why
# every call above could omit the argument entirely.
#
# An operator has two slots and must say which, rather than have a default
# quietly pick one. Lowering slot 0 of $P^i{}_j$ gives the associated bilinear
# form $P_{ij} = g_{ik}P^k{}_j$ --- and because this operator is self-adjoint,
# that form comes out **symmetric**. Self-adjointness of the map and symmetry of
# the lowered form are the same statement.

# %%
try:
    Proj_B.lower()  # two slots: refuses to guess
except FrameMismatch as err:
    print("Proj_B.lower() ->", err)

P_form = Proj_B.lower(slot=0)  # (up, down) -> (down, down)
print(
    "\nlowered slots:",
    (P_form.row, P_form.col),
    "  symmetric:",
    P_form.mat.is_symmetric(),
)
assert P_form.mat.is_symmetric()
P_form

# %% [markdown]
# ### Saying where you want to end up, instead of what to do
# `lower(slot=0)` names an *operation*. The alternative is to name the
# *destination* --- which is how OGRePy
# ([Shoshany 2024](https://github.com/bshoshany/OGRePy)) does it: a tensor there
# is asked for a representation, `indices=(1, -1)`, and the package works out
# which indices to raise or lower. Nothing has to identify a slot, because going
# from $(\text{up}, \text{down})$ to $(\text{down}, \text{down})$ already says
# "lower the first index."
#
# `with_variance` is that form, taking one variance per slot as `'up'`/`'down'`
# or OGRePy's $\pm 1$. Slots already in the requested variance are left alone.
# Both spellings stay useful: `with_variance` when you want the result,
# `lower`/`raise_` when the step itself is the point --- which, in a notebook
# about keeping variance straight, it often is.

# %%
print("Proj_B.variance =", Proj_B.variance)
assert Proj_B.with_variance("down", "down").mat == P_form.mat  # same as lower(slot=0)
assert Proj_B.with_variance(-1, -1).mat == P_form.mat  # OGRePy-style +1/-1
assert Proj_B.with_variance(1, -1).mat == Proj_B.mat  # already there: a no-op
Proj_B.with_variance("down", "down")

# %% [markdown]
# The metric is the neatest case. Raising *both* of its indices --- moving
# $g_{ij}$ from $V^\ast \otimes V^\ast$ to $V \otimes V$ --- is exactly the
# inverse metric $g^{ij}$, and `with_variance` gets there without being told
# that inversion is what raising a metric means.

# %%
g_lower = TMatrix.metric_of(Bway)
g_upper = g_lower.with_variance(1, 1)
assert g_upper.mat == sp.Matrix(Bway.metric).inv()
print("g_ij =", g_lower.mat.tolist(), "  ->  g^ij =", g_upper.mat.tolist())
g_upper

# %%
print("self-adjoint:", sp.simplify(Proj_B.adjoint().mat - Proj_B.mat).is_zero_matrix)
print("symmetric:", Proj_B.T.mat == Proj_B.mat)
assert Proj.in_frame(Bway).mat == Proj_B.mat
assert (Proj * home).in_frame(Bway).mat == (Proj_B * home_B).mat
print("nearest point on Broadway to 7th & C:", (Proj * home).mat.T, "in E")
Proj, Proj_B

# %% [markdown]
# ### Declaring a frame orthonormal
# The old demo made this point by handing `change_of_basis` a shear between two
# bases both declared Euclidean and watching it be refused. With $P$ derived,
# that failure has nowhere left to happen --- so the check moves to where the
# false claim is actually made, the frame constructor.

# %%
th = sp.Symbol("theta", real=True)
R = City.frame(
    "R", [[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]], orthonormal=True
)
print("rotation accepted as an orthonormal frame of City:", R)
try:
    City.frame("S", [[1, sp.Rational(3, 2)], [0, 1]], orthonormal=True)
except FrameMismatch as err:
    print("shear rejected ->", str(err)[:88], "...")

# %% [markdown]
# ### Metric vs. inner product: the degenerate rung
# A metric only has to be symmetric and *non-degenerate* for $\flat$ and
# $\sharp$ to be isomorphisms; positive-definiteness is the stronger condition
# that makes it an inner product proper. Below both is the degenerate case
# ($\det g = 0$) --- legitimate geometry, this is kingdon's `Algebra(p, q, r)`
# with $r > 0$, but $\flat$ is no longer invertible so indices cannot be raised.
# `typed_matrix` accepted such a metric and then died inside sympy with
# `NonInvertibleMatrixError`; here it is caught and named.

# %%
Degen = VectorSpace(
    "D", 2, metric=sp.Matrix([[1, 0], [0, 0]])
)  # e2 is a null direction
d = Degen.frame("d")
print("degenerate:", Degen.is_degenerate, "| lowering still works:", d.metric.tolist())
try:
    TMatrix.covector([1, 0], d).raise_()
except FrameMismatch as err:
    print("raising ->", err)

# %% [markdown]
# ## The payoff: adjoints of maps between *different* spaces
# `typed_matrix.adjoint` refused anything but an endomorphism --- with the metric
# stored per-basis there was no way to ask for "the other space's $g$." That is
# the one piece of `change-basis-no-numpy.ipynb` that had no code: the
# coordinate-free transpose, the pullback $f^\ast : W^\ast \to V^\ast$.
#
# Give the city a third dimension --- elevation --- with a metric that makes
# climbing cost twice as much per unit as walking: $g_V = \mathrm{diag}(1,1,4)$.
# Let $f$ flatten a hillside onto the map, with elevation shifting you east as
# you climb. Then $f^\dagger$ is the unique map with
# $$g_V(f^\dagger w, u) = g_W(w, f u) \qquad \text{for all } u \in V,\ w \in W$$
# and it is emphatically *not* the transpose.

# %%
Hill = VectorSpace("V", 3, metric=sp.diag(1, 1, 4))  # east, north, up (climbing costs)
v = Hill.frame("v")

A = sp.Matrix([[1, 0, 1], [0, 1, 0]])  # 2x3: climbing also carries you east
fmap = TMatrix.map(A, source=v, target=E)  # f: Hill -> City
fdag = fmap.adjoint()  # f†: City -> Hill

print("f  slots:", (fmap.row, fmap.col), "shape", fmap.mat.shape)
print("f† slots:", (fdag.row, fdag.col), "shape", fdag.mat.shape)
print("f†             =", fdag.mat.tolist())
print("bare transpose =", A.T.tolist())
fmap, fdag

# %%
# Verified for a *generic* u and w, so this is a proof and not a sample.
u3 = TMatrix.vector(sp.symbols("u_1 u_2 u_3", real=True), v)
w2 = TMatrix.vector(sp.symbols("w_1 w_2", real=True), E)
lhs = (fdag * w2).T.lower() * u3  # g_V(f† w, u)
rhs = w2.T.lower() * (fmap * u3)  # g_W(w, f u)
print("g_V(f†w, u) =", sp.expand(lhs))
print("g_W(w, f u) =", sp.expand(rhs))
assert sp.simplify(lhs - rhs) == 0

# %%
# And the adjoint is geometry, not bookkeeping: computing it in the Broadway
# frame gives the same map as computing it in the grid frame and changing frame.
# Note in_frame() touched only the City slot -- the Hill slot is another space.
f_B = fmap.in_frame(Bway)
print("f in the Broadway frame:", f_B.mat.tolist(), (f_B.row, f_B.col))
assert sp.simplify(f_B.adjoint().mat - fdag.in_frame(Bway).mat).is_zero_matrix
print("adjoint commutes with change of frame")

# %% [markdown]
# ## What the split bought
# - **The metric is stated once.** A frame's Gram matrix is computed, so two
#   frames of one space cannot encode different geometries. The old
#   `gB = Bmat.T * Bmat` line and its cautionary comment are gone; the type does
#   the insisting.
# - **`change_of_basis` takes no matrix.** $P = B_t^{-1}B_s$ is derived, so an
#   inconsistent change of frame is unrepresentable rather than rejected. The
#   claim that *was* being checked --- "this frame is orthonormal" --- moved to
#   where it is made.
# - **`adjoint` generalizes to $f : V \to W$**, which is what makes the pullback
#   material from `change-basis-no-numpy.ipynb` expressible at all.
# - **Errors can name the actual problem**, including the case the old design
#   could not even represent: two frames of the *same* space.
#
# The cost is ceremony --- `VectorSpace` then `Frame` before you can write a
# single matrix. `Frame.euclidean("E", 2)` papers over it for one-off examples,
# at the price of minting an anonymous space each time, so two frames built that
# way can never be related. That is the right default: silence about which space
# you are in was exactly the old bug.
