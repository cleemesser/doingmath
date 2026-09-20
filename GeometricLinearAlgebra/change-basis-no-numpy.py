# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: doingmath (3.14.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Change of basis
# The purpose of this notebook, is to illustrate the concepts that arise with change of coordinate system.
# This brings up the concept of the of the dual map, its relationship to the transpose operation on matrices.
#
# This leads to the introduction of the metric transformation which "raises" and "lowers" indices
#
# **This is the numpy-free variant of `change-basis.py`:** every number is either
# a `kingdon` multivector (coordinate free) or a `sympy` expression (attached to a
# basis). Nothing is a float, so every identity below is checked *exactly*.
#
#


# %%
import kingdon
from kingdon import Algebra

# widget related stuff
import plotly.graph_objects as go

from IPython.display import display  # no clear_output needed -- see above
import ipywidgets
import wigglystuff
import clmmathtools.viz as mv
from clmmathtools.viz import show_md as md, show_expr

import sympy as sp
from sympy import Matrix, latex, sqrt

sp.init_printing()
# %% [markdown]
# We will use the `kingdon` library to work with vectors in coordinate free way. We use sympy to work with matrices attached to specific coordinate frames.
#
# `kingdon` carries whatever coefficient type you hand it, so a multivector built
# from `sympy` expressions stays symbolic: `(e1 - e2)/sqrt(2)` really does square
# to $1$, not to $0.9999999999999998$.

# %%
# how kingdon defines 2D vector space
alg2 = Algebra(2, 0, 0)  # $\mathbb{R}^2$
e1, e2 = (
    alg2.blades.e1,
    alg2.blades.e2,
)  # import kingdon's defintions for standard defintion
md(f"we can use kingdon's standard basis: ${latex((e1, e2))}$")
e1, e2
# locals().update(alg2.blades) # add defintions for e, e1,e2,e12 to global space

# %% [markdown]
# ### Two small bridges between `kingdon` and `sympy`
# `scalar` pulls the grade-0 part of a multivector out as a *simplified sympy
# scalar*; `grade1_to_matrix` turns a grade-1 multivector into a sympy column
# vector of its components. These two functions are the only place coordinates
# get introduced -- everything upstream of them is coordinate free.
#
# `same` replaces `np.allclose`: with exact arithmetic we can ask whether the
# difference *is* zero rather than whether it is small.


# %%
def scalar(mvec: kingdon.MultiVector):
    """The scalar (grade-0) part of a multivector, as a simplified sympy expression."""
    return sp.simplify(sp.sympify(mvec.e))


def grade1_to_matrix(mvec: kingdon.MultiVector) -> Matrix:
    """The components of a grade-1 multivector as a sympy column vector."""
    return Matrix([sp.sympify(mvec.e1), sp.sympify(mvec.e2)])


def same(a, b) -> bool:
    """Exact equality of two sympy matrices/scalars, after simplification."""
    d = sp.sympify(a) - sp.sympify(b)
    d = d.applyfunc(sp.simplify) if isinstance(d, sp.MatrixBase) else sp.simplify(d)
    return d.is_zero_matrix if isinstance(d, sp.MatrixBase) else d == 0


def sc(x) -> str:
    """LaTeX for an inline scalar (exact, so no format spec needed)."""
    return latex(sp.simplify(sp.sympify(x)))


# %% [markdown]
# We define a new set of basis vectors which are NOT orthonormal
# %%
# lets define our new basis vectors in terms of the standard basis
newbasis_list = [
    [1, 0],  # same as e1
    [1, 1],
]  # length sqrt(2), at 45 degrees to e1 (note: g_22 = |b_2|^2 = 2)

b1_mv = alg2.vector(newbasis_list[0])
b2_mv = alg2.vector(newbasis_list[1])
# newbasis_mv = alg2.vector(newbasis) broadcasted version
mv.show_md(f"""Let's define our new coordinate system with basis vectors
b1_mv, b2_mv:
$$b_1 = {b1_mv} \\\\
 b_2 = {b2_mv}$$""")

mv.show_md(
    "$b_1$ and $b_2$ when in expressed in matrix form in the standard basis are: "
)
Matrix(newbasis_list[0]), Matrix(newbasis_list[1])
# %% [markdown]
# We can put the basis column vectors together to form a matrix $B$

# %%
B = Matrix(newbasis_list).T
# could also make this with
# Matrix.hstack(Matrix(newbasis_list[0]),Matrix(newbasis_list[1]))
md("$B = $")
B

# %%
md(r"""Note that the B matrix from the point of view of vector components expressed in
the $\{b_i\}$ basis, will transform (via a shear operation) into vector components in the standard
 basis""")
md(
    r"""$B\,\vec{b}_2 = """
    rf"""{latex(B)}{latex(Matrix([0, 1]))} = {latex(B * Matrix([0, 1]))}$"""
)

# %%
md("""We can visualize the standard basis vectors and superimpose the new
basis vectors on the graph. Note that can move $e_2$ to $b_2$ with a shearing
operation by "pushing" it to the right.
""")
if mv.backends.in_notebook():  # NB: in_notebook is a function -- must be called
    plane1 = mv.Plane()
else:
    plane1 = mv.Plane(backend="vedo")  # for use in python script

# clmmathtools coerces whatever it is given with np.asarray(..., float), so plain
# lists and sympy Matrices/expressions all work -- numpy stays inside clmmathtools.
plane1.vector(
    newbasis_list[0], label="$b_1=e_1$", color=mv.GREEN
)  # bump index from 0 to 1 ?
plane1.vector(newbasis_list[1], label="$b_2$", color=mv.RED)
plane1.vector([0, 1], label="$e_2$", color=mv.PURPLE)
plane1.display()

# %%
md(r"""it follows that $B^{-1}$ should take components in the standard basis to
components in the new basis.

$B^{-1} =$""")
Binv = B.inv()
Binv
# %%
# let's define some other vectors in
v_mv = 3 * e1 + e2
u_mv = (e1 - e2) / sqrt(2)  # length 1, exactly

# A *generic* vector: instead of drawing random floats we give w symbolic
# components, which upgrades every check below from "true for one sample" to
# "true for all w".
w1_sym, w2_sym = sp.symbols("w_1 w_2", real=True)
# alg2.vector(name="w",keys=('e1','e2'))
w_mv = alg2.vector([w1_sym, w2_sym])
v_mv, u_mv, w_mv

# %% [markdown]
# ### finding the componets of the $u$ and $v$ and getting their inner product
# #### Important note: component index positions
# when working with non-orthogonal coordinate systems there is a convention to put indices in different positions as the components may be attached to two types of vectors relative to the basis vector space $V$.
#
# If the components are attached to the base space $V$, such that $\vec{v}$ = $\sum v^i \vec{b}_i$ where as when giving components of the dual space $\vec{v}^* = \sum_i v_i \vec{b}^*_i$
#
# As a short hand, physicists will often drop the $*$ and just use upper and lower indices to indicate the vectora and component type. So the i-th dual basis vector
# $$
# \vec{b}^i \triangleq \vec{b}^*_i
# $$

# %%
u_arr = grade1_to_matrix(u_mv)  # components of u in standard basis
v_arr = grade1_to_matrix(v_mv)  # components of v in standard basis
w_arr = grade1_to_matrix(w_mv)
md(r"components $u^{i} = " rf"{latex(u_arr)}$")
md(r"components $v^{i} = " rf"{latex(v_arr)}$")
md(r"$<\vec{u},\vec{v}> = \sum_i u^i v^i = " rf"{sc(u_arr.dot(v_arr))}$")
md(r"$<\vec{w},\vec{v}> = \sum_i w^i v^i = " rf"{sc(w_arr.dot(v_arr))}$")
md(r"$<\vec{w},\vec{u}> = \sum_i w^i u^i = " rf"{sc(w_arr.dot(u_arr))}$")

# %% [markdown]
# ## Defining the gram matrix G
# attached to new basis because it is not orthogonal.
# Once we move the components of $\vec{u},\vec{v}$ into the new basis
# ($u' = B^{-1}u$, $v' = B^{-1}v$), the inner product is no longer a plain sum --
# it picks up the Gram matrix:
# $$\langle \vec{u},\vec{v}\rangle = u'^{T} G_b\, v' = u'^i (G_b)_{ij} v'^j$$
# Note this is $G_b$ with both indices *down*, not its inverse: two upper-index
# component vectors need two lower indices to contract with.
#
# ### create Gram matrix for new basis "b"
# usual convention $\vec{v} = v^i b_i = v_j b^j$ (assuming einstein sum convention)
# set of $b_i$ are the basis of $V$ and the $b^i$ are the dual basis of $V^*$
# $$b^j(b_i) = \delta^j_i$$
#
# $g$ defines a natural isomorphism (map) between $V$ and $V^*$
# $$G_{ij} = \langle b_i, b_j \rangle \quad \text{for a given choice of basis}$$
# for the standard basis of $\mathbb{R}^n$, where the bases are orthonormal, $G_{ij} = \delta_{ij}$ so the components of "columns" $v^i$ are unchanged when they become "rows" via the mechanical transpose.
#
# but in our new basis, $b_i$ for i=1 to 2. is not orthonormal. Let's calculate this below. I will call the matrix `Gb`

# %%
Gb = B.T @ B
Gb_lower = Gb
Gb_raise = Gb_lower.inv()
Gb_lower, Gb_raise

# %% [markdown]
# The same Gram matrix built the coordinate-free way, straight out of the
# `kingdon` inner products $\langle b_i, b_j \rangle$ -- no matrix $B$ involved.

# %%
basis_mvs = (b1_mv, b2_mv)
Gb_free = Matrix(2, 2, lambda i, j: scalar(basis_mvs[i] | basis_mvs[j]))
md(rf"$\langle b_i, b_j \rangle = {latex(Gb_free)}$")
assert same(Gb_free, Gb), (Gb_free, Gb)
md(r"verified exactly: the coordinate-free Gram matrix equals $B^T B$")

# %%
mv.show_md(f"$Gb_{{ij}} = {latex(Gb_lower)}$")
mv.show_md(f"$Gb^{{ij}} = {latex(Gb_raise)}$")

# %%
Gb_lower @ Gb_raise  # confirm that they are inverses

# %%
w_mv

# %% [markdown]
# ### Checking the gram matrix
# - calculate the inner products and lengths of vectors in coord free and standard basis
# - these should be invariant under change of basis so we can check inner product in transformed coordinates and they should be the same if $g$ is correct

# %%
inner_prod_uv = scalar(u_mv | v_mv)
md(r"$\langle u_{mv}, v_{mv}\rangle = " rf"{sc(inner_prod_uv)}$ (coordinate free)")
md(r"$\sum_i u^i v^i = " rf"{sc(u_arr.dot(v_arr))}$ (standard basis)")
assert same(inner_prod_uv, u_arr.dot(v_arr))

# now change to newbasis
u_arr_b = Binv @ u_arr
v_arr_b = Binv @ v_arr
w_arr_b = Binv @ w_arr
mv.show_md(
    r"$ub = B^{-1} u$ to transforms compoents of $\vec{u}$ in std basis to components in new basis $b_j$"
)
mv.show_md(r"$ub = " rf"{latex(u_arr_b)}_b$")
mv.show_md(r"$vb = " rf"{latex(v_arr_b)}_b$")
mv.show_md(
    r"where as the coord free versions are "
    r"$\vec{u} = $"
    rf" ({u_mv})"
    r" and $\vec{v} = $"
    rf"({v_mv})"
)

# %%
md(r"check that $ub^1 b_1 + ub^2 b_2 = \vec{u}$:")
u_recon = u_arr_b[0] * b1_mv + u_arr_b[1] * b2_mv
assert same(grade1_to_matrix(u_recon), u_arr)
u_recon, u_mv

# %%
md(r"same check for the *generic* vector $w$ -- this one holds for all $w_1, w_2$:")
w_recon = w_arr_b[0] * b1_mv + w_arr_b[1] * b2_mv
assert same(grade1_to_matrix(w_recon), w_arr)
w_recon, w_mv

# %%
mv.show_md(r'now show that the naive summation "dot" product does not work')
mv.show_md(
    r"$\langle u,v\rangle$ = "
    rf"{sc(inner_prod_uv)}"
    r" $\ne \sum_i ub^i\, vb^i$ = "
    rf"{sc(u_arr_b.dot(v_arr_b))}"
)
assert not same(inner_prod_uv, u_arr_b.dot(v_arr_b))

# %%
# ...but contracting through the metric does
quad_uv = (u_arr_b.T @ Gb_lower @ v_arr_b)[0, 0]
quad_uv, same(quad_uv, inner_prod_uv)

# %%
# the same for the generic w: |w|^2 in the b-basis vs coordinate free
quad_ww = sp.expand((w_arr_b.T @ Gb_lower @ w_arr_b)[0, 0])
quad_ww, scalar(w_mv**2), same(quad_ww, scalar(w_mv**2))

# %%
(Gb_lower @ w_arr_b).dot(w_arr_b), scalar(w_mv**2)
assert same((Gb_lower @ w_arr_b).dot(w_arr_b), scalar(w_mv**2))

# %%
# get back originals: lowering u's index in the b basis
mv.show_md(
    rf"covector components $ub_i$ in basis b: ${latex((Gb_lower @ u_arr_b).T)}_b$"
)

# %%
# <u, v> again -- G is symmetric, so it does not matter which side we lower
same((Gb_lower @ u_arr_b).dot(v_arr_b), u_arr_b.dot(Gb_lower @ v_arr_b))

# %%
v_arr_b  # [2,1] vector compoents (upper index)
v_arr_b_lower = Gb_lower @ v_arr_b
v_arr_b_lower  # [3,4]
# %%
(v_arr_b.T @ Gb_lower @ v_arr_b)[0, 0], scalar(v_mv**2)
# %%
v_arr_b.T, v_arr_b.T @ Gb_lower, v_arr_b.T @ Gb_lower @ v_arr_b

# %% [markdown]
# ### Construct the dual basis vectors for basis $b_i$
#

# %%
b1_arr = grade1_to_matrix(b1_mv)  # b1 in std coords
b2_arr = grade1_to_matrix(b2_mv)
b1_arr_b = Matrix([1, 0])  # b1 in B basis
b2_arr_b = Matrix([0, 1])  # b2 in B basis
b1_dual_b = Gb_raise @ b1_arr_b
b2_dual_b = Gb_raise @ b2_arr_b
b1_dual_b, b2_dual_b
# b1_dual.T @ b1_arr, b1_dual @b2_arr

# %%
md(
    rf"""In the b_i coordinates, the components for the dual basis are: ${latex(b1_dual_b.T)}$ and ${latex(b2_dual_b.T)}$"""
)
Gb_raise

# %% [markdown]
# The dual basis vectors themselves, coordinate free: the rows of $G_b^{-1}$ tell
# us which combination of $b_1, b_2$ each $b^i$ is.

# %%
b1_dual_mv = b1_dual_b[0] * b1_mv + b1_dual_b[1] * b2_mv  # top row of Gb_raise
b2_dual_mv = b2_dual_b[0] * b1_mv + b2_dual_b[1] * b2_mv  # bottom row of Gb_raise
b1_dual_mv, b2_dual_mv, b1_mv, b2_mv

# %% [markdown]
# Check all four entries of $\langle b^i, b_j \rangle = \delta^i_j$, first
# coordinate-free with `kingdon`, then via the Gram matrix in the b-basis.

# %%
dual_mvs = (b1_dual_mv, b2_dual_mv)
delta_mv = Matrix(2, 2, lambda i, j: scalar(dual_mvs[i] | basis_mvs[j]))
assert delta_mv == sp.eye(2), delta_mv
md(rf"coordinate free: $\langle b^i, b_j \rangle = {latex(delta_mv)} = \delta^i_j$")

# %%
dual_bs = (b1_dual_b, b2_dual_b)
basis_bs = (b1_arr_b, b2_arr_b)
delta_b = Matrix(2, 2, lambda i, j: (dual_bs[i].T @ Gb_lower @ basis_bs[j])[0, 0])
assert delta_b == sp.eye(2), delta_b
md(rf"in the b-basis: $(b^i)^k (G_b)_{{kl}} (b_j)^l = {latex(delta_b)} = \delta^i_j$")

# %% [markdown]
# ### When can we skip the metric?
# `b1_dual_b` holds the *contravariant* (upper-index) components of $\vec{b}^1$,
# so pairing it with $v^i$ still needs $G_b$ -- the naive dot product is wrong:
# $b^1 \cdot b_2 \to -1$ where the true answer is $0$, and $b^1 \cdot b^1 \to 5$
# where $\lVert b^1 \rVert^2 = 2$.
#
# But *lowering* the dual basis gives back $\delta$ exactly, and then the
# contraction is a plain sum with no matrix at all. That is the real payoff of
# the index bookkeeping: matched up/down indices contract for free; only
# same-position indices need $g$.

# %%
md(
    "naive dot of upper-index components (WRONG): "
    rf"$$\sum_i (b^1)^i  (b_2)^i = {sc(b1_dual_b.dot(b2_arr_b))}$$, "
    rf"$$\sum_i (b^1)^i (b^1)^i = {sc(b1_dual_b.dot(b1_dual_b))}$$"
)
md(
    r"true values via $G_b$: "
    rf"$\langle b^1, b_2 \rangle = {sc((b1_dual_b.T @ Gb_lower @ b2_arr_b)[0, 0])}$, "
    rf"$\lVert b^1 \rVert^2 = {sc((b1_dual_b.T @ Gb_lower @ b1_dual_b)[0, 0])}$"
)

# lowering the dual basis recovers delta, so the pairing needs no metric
b1_dual_lower = Gb_lower @ b1_dual_b
b2_dual_lower = Gb_lower @ b2_dual_b
md(
    rf"lowered: $(b^1)_i = {latex(b1_dual_lower)}$, "
    rf"$(b^2)_i = {latex(b2_dual_lower)}$ -- these are just $\delta$"
)
md(
    r"so the metric-free pairing works: "
    rf"$(b^1)_i (b_2)^i = {sc(b1_dual_lower.dot(b2_arr_b))}$, "
    rf"$(b^1)_i (b_1)^i = {sc(b1_dual_lower.dot(b1_arr_b))}$"
)

# %%
md("""Now plot the basis $\\{b_1, b_2\\}$ together with its dual basis
$\\{b^1, b^2\\}$. If we identify $V^*$ with $V$ (which the inner product lets us
do) both live in the same picture. Note the defining geometry: $b^1$ is
perpendicular to $b_2$, and $b^2$ is perpendicular to $b_1$ -- each dual vector
is orthogonal to the *other* basis vector, and scaled so its projection onto its
own partner is exactly 1. Shear $b_2$ further right and $b^1$ swings further
down to keep that perpendicularity: the dual basis moves *opposite* to the
basis, which is what the up/down index convention is tracking.
""")
if mv.backends.in_notebook():
    plane1 = mv.Plane()
else:
    plane1 = mv.Plane(backend="vedo")  # for use in python script

plane1.vector(
    newbasis_list[0], label="$b_1=e_1$", color=mv.GREEN
)  # bump index from 0 to 1 ?
plane1.vector(newbasis_list[1], label="$b_2$", color=mv.RED)
# plane1.vector([0, 1], label='$e_2$', color=mv.PURPLE)
# sympy column Matrices go straight into clmmathtools -- it floats them internally
plane1.vector(
    grade1_to_matrix(b1_dual_mv), label="$b^1$ (dual)", color=mv.BLUE
)  # perpendicular to b_2
plane1.vector(
    grade1_to_matrix(b2_dual_mv), label="$b^2 = e_2$ (dual)", color=mv.YELLOW
)  # perpendicular to b_1
plane1.display()

# %% [markdown]
# ### Notes on going numpy-free
# - **`kingdon` is agnostic about its coefficients.** Feed it sympy and the
#   multivectors stay symbolic, so `u_mv**2` is exactly `1` and `w_mv**2` comes
#   out as $w_1^2 + w_2^2$ for a *generic* $w$. The checks above are therefore
#   proofs, not samples, and `assert same(...)` replaces `np.allclose(...)`.
# - **`clmmathtools` still uses numpy internally, and that is fine.** `Plane.vector`
#   ends with `np.asarray(vec, float).reshape(2)`, and sympy matrices/expressions
#   convert cleanly through `float()`. So plots work with sympy column vectors,
#   plain lists, or kingdon components -- the notebook never imports numpy.
#   The one requirement is that whatever you plot be *numeric*: a symbolic
#   vector like `w_mv` has to be `.subs(...)`-ed to numbers before it can be drawn.
# - `einops.einsum` was only doing $u^i G_{ij} v^j$; in sympy that is just
#   `(u.T @ G @ v)[0, 0]`, and `Matrix.dot` covers the plain sums.
# - The cost is speed (sympy `simplify` on every scalar) -- irrelevant at 2x2,
#   but this pattern would not scale to a grid of thousands of vectors.

# %% [markdown]
# ### Vectors and their dual vectors, also known as covectors, linear forms
# For a vector space $V$ endowed with an inner product. Let's consider vector $u,v \in V$ and their dual vectors $u^*,v^* \in V^*$ and $L$ linear function: $V \rightarrow V$:
# $$
#   v^*(v) = v \cdot v = <v,v>$$
# $$
#   v^*(u) = v \cdot u = <v,u>
# $$
# $$
#   v^* (L u) = (L^*(v^*))(u)
# $$
# given that we have this inner product is defined, this is also called the adjoint map.
# this can be written as an example of a pull back operation: $L^*(\phi) = \phi \circ L$
# this can be generalized for $f:V \rightarrow W$, $f^*:W^* \rightarrow V^*$
# this is called pre-composition
#
# we have just defined the coordinate free defintion of the "mechanical
#  transpose" for matrices in orthogoncal basis coordinates
#
# to see how this coordinate-free pullback yields the traditional matrix
# transpose, we introduce arbitrary bases:Let $B_V = \{v_1, v_2, \dots, v_n\}$
# be a basis for $V$. Let $B_W = \{w_1, w_2, \dots, w_m\}$ be a basis for $W$.
#
# Every choice of basis has a unique dual basis. We construct:$B_V^* = \{v^1, v^2,\dots, v^n\}$ for $V^{*}$, defined by $v^i(v_j) = \delta^i_j$ (where
# $\delta^i_j = 1$ if i=j, and 0 otherwise).$B_W^* = \{w^1, w^2, \dots,
# w^m\}$ for $W^{*}$, defined by $w^i(w_j) = \delta^i_j$.
#
# [authors note: this is confusing because sometimes use components and sometimes use index for vectors themselves. May need to put hats or boldface them to distinguish]
#
# The matrix A representing the forward map f relative to these bases is found
# by looking at where f sends the basis vectors of V, expressed in terms of the
# basis vectors of W, $\{b_i \}$ (note switch from $w_i$, the set $b_i$ are the basis of W and b^{i} are the "covariant basis" of $W^*$. Note in $W^*$, $b^i$ are just as invariant as any other vector in $W^*$ but they "co-vary" with changes in $W$: $b^i(b_j) = \delta^{i}_{j}$ note I'm not sure if this has flipped up and down
# $$
# f(v_{j})=\sum_{i=1}^{m}A_{ij}b_{i}
# $$
# $$
# <b^i , f(v_j)> = A_{ij}
# $$
#
# #### The Matrix Representation of $f^{*}$
# Now let's find the matrix B that represents the pullback map $f^{*}: W^{*} \to V^*$. Because $f^{*}$ goes from $W^{*}$ to $V^{*}$, it acts on the basis vectors of $W^{*}$ and
# expresses them in terms of the basis vectors of $V^{*}$: $f^{*}(b^{i})=\sum
# _{j=1}^{n}B_{ji}v^{j}$ To extract the specific entry $B_{ji}$, we evaluate
# this functional on a basis vector $v_{j}$: $\langle f^{*}(b^{i}),v_{j}\rangle =B_{ji}$
#
# #### Putting things together
#
# We now evaluate the coordinate-free definition of the pullback using our basis
# elements $b^{i}$ and $v_{j}$:
# $$
# \begin{aligned}\text{By\ definition\ of\
# pullback:}\quad \langle f^{*}(b^{i}),v_{j}\rangle &=\langle
# b^{i},f(v_{j})\rangle \\ \text{Substitute\ our\ matrix\ entries\ }A\text{\
# and\ }B:\quad \quad \quad \quad B_{ji}&=A_{ij}\end{aligned}
# $$
# Because $B_{ji}= A_{ij}$, the matrix B representing the dual map is the row-column swap of the matrix A.
#
# ### Summary
# The traditional matrix transpose is not just an arbitrary
# rule about flipping rows and columns. It is the algebraic consequence of
# evaluating a coordinate-free pullback (the dual map) through the lens of dual
# bases.
