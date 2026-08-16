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
#     display_name: doingmath (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# The purpose of this notebook, is to illustrate the concepts of the dual map, its relationship to the transpose operation on matrices.
# this leads to the concept of the metric tensor which "raises" and "lowers" indices
# for a vector $u,v \in V$ and their dual vectors $u^*,v^* in V^*$ and $L$ linear function: $V->V$:
#   $v^*(v) = v \cdot v = <v,v>$
#   $v*^(u) = v \cdot u = <v,u>$
#   $v^* (L u) = (L^*(v^*))(u) =
# given that we have this inner product is defined, this is also called the adjoint map.
# this can be written as an example of a pull back operation: $L^*(\phi) = \phi \circ L$
# this can be generalized for $f:V \rightarrow W$, $f^*:W^* \rightarrow V^*$
# this is called pre-composition
#
# we have just defined the coordinate free defintion of the "mechanical
#  transpose" for matrices in orthogoncal basis coordinates
#
# to see how this coordinate-free pullback yields the traditional matrix
# transpose, we introduce arbitrary bases:Let \(B_V = \{v_1, v_2, \dots, v_n\}\)
# be a basis for V.Let \(B_W = \{w_1, w_2, \dots, w_m\}\) be a basis for W.Every
# choice of basis has a unique dual basis. We construct:\(B_V^* = \{v^1, v^2,
# \dots, v^n\}\) for \(V^{*}\), defined by \(v^i(v_j) = \delta^i_j\) (where
# \(\delta^i_j = 1\) if i=j, and 0 otherwise).\(B_W^* = \{w^1, w^2, \dots,
# w^m\}\) for \(W^{*}\), defined by \(w^i(w_j) = \delta^i_j\).
#
#
# The matrix A representing the forward map f relative to these bases is found
# by looking at where f sends the basis vectors of V, expressed in terms of the
# basis vectors of W, $\left{b_i \right}$ (note switch from $w_i$, the set b_i are the basis of W and b^{i} are the "covariant basis" of $W^*$. Note in $W^*$, $b^i$ are just as invariant as any other vector in $W^*$ but they "co-vary" with changes in $W$: $b^i(b_j) = \delta^{i}_{j}$ note I'm not sure if this has flipped up and down
# $$
# f(v_{j})=\sum_{i=1}^{m}A_{ij}b_{i}
# $$
# $$
# <b^i , f(v_j)> = A_{ij}
# $$
#
# 4. The Matrix Representation of \(f^{*}\)Now let's find the matrix B that
# represents the pullback map \(f^*: W^* \to V^*\). Because \(f^{*}\) goes from
# \(W^{*}\) to \(V^{*}\), it acts on the basis vectors of \(W^{*}\) and
# expresses them in terms of the basis vectors of \(V^{*}\):\(f^{*}(b^{i})=\sum
# _{j=1}^{n}B_{ji}v^{j}\)To extract the specific entry \(B_{ji}\), we evaluate
# this functional on a basis vector \(v_{j}\):\(\langle
# f^{*}(b^{i}),v_{j}\rangle =B_{ji}\)
#
# 5. The Grand Synthesis
#
# We now evaluate the coordinate-free definition of the pullback using our basis
# elements \(b^{i}\) and \(v_{j}\):\(\begin{aligned}\text{By\ definition\ of\
# pullback:}\quad \langle f^{*}(b^{i}),v_{j}\rangle &=\langle
# b^{i},f(v_{j})\rangle \\ \text{Substitute\ our\ matrix\ entries\ }A\text{\
# and\ }B:\quad \quad \quad \quad B_{ji}&=A_{ij}\end{aligned}\)Because \(B_{ji}
# = A_{ij}\), the matrix B representing the dual map is the row-column swap of
# the matrix A.SummaryThe traditional matrix transpose is not just an arbitrary
# rule about flipping rows and columns. It is the algebraic consequence of
# evaluating a coordinate-free pullback (the dual map) through the lens of dual
# bases.


# %%
import kingdon
from kingdon import Algebra
import numpy as np
import einops as ein
from einops import einsum

# widget related stuff
import plotly.graph_objects as go

from IPython.display import display  # no clear_output needed -- see above
import ipywidgets
import wigglystuff
import mathviz as mv


import sympy as sp
from sympy import Matrix
sp.init_printing()
# %%
alg2 = Algebra(2,0,0) # $\mathbb{R}^2$
locals().update(alg2.blades) # add defintions for e1,e2,e12 to global space
# %%
alg2.blades
# %% [markdown]
# We define a new set of basis vectors which are NOT orthonormal
# %%
newbasis = np.array([[1,0], [1,1]], dtype=np.float64)
sp.Matrix(newbasis[0]), sp.Matrix(newbasis[1])
# %%
if mv.backends.in_notebook:
    plane1 = mv.Plane()
else:
    plane1 = mv.Plane(backend='vedo') # for use in python script

plane1.vector(newbasis[0],label='b1',color=mv.GREEN) # bump index from 0 to 1 ?
plane1.vector(newbasis[1],label='b2', color=mv.RED)
plane1.display(interactive=True)
# %%
newbasis @ np.array([1,0]).T
# %%
# ein.einsum(basis,
# %%
b1_mv = alg2.vector(newbasis[0])
b2_mv = alg2.vector(newbasis[1])
newbasis_mv = alg2.vector(newbasis)

# %%
newbasis_mv

# %%
v_mv = 3*e1 + e2
u_mv = (e1 - e2)/np.sqrt(2)  # length 1
w_mv = e1
# %%
# %%

# %%
# toscalar = lambda mvector: mvector.values()[0]
# toscalar = lambda mvector: mvector.values()
def toscalar(mvec):
    """take the scalar portion of a multi vector into a regular float
    """
    return mvec.e
    # g0 = mvec.grade(0)
    #vals = g0.values()
    #print(g0, vals)
    #if vals:
    #    #print(f'found it: {vals[0]}')
    #    r = vals[0]
    #else:
    #    r = 0
    #return r

def grade1tovector2(mvec: kingdon.MultiVector):
    return np.array((mvec.e1, mvec.e2),dtype=np.float64)
# %%
# change of basis that turns {e_i} basis into {b_i} coords basis
#   A' = S^{-1} A S or S A S^{-1}, figure out which
# Convention: use capitals for numpy matrices

S = [[b1_mv.e1, b2_mv.e1], [b1_mv.e2, b2_mv.e2]]
print([[type(xx) for xx in row] for row in S])
S = np.array(S)
Sinv = np.linalg.inv(S)
sp.Matrix(S), sp.Matrix(Sinv)
# %% [markdown]
# ## metric for the new coordinate system
# $g$ and $G$
# g(u,v): u,v -> R

# %%
u_arr = grade1tovector2(u_mv)
v_arr = grade1tovector2(v_mv)
np.dot(u_arr,v_arr)

# %% [markdown]
# ## "Metric G, Ginv"
# attached to new basis because it is not orthogonal
# with matrix representations of $u_arr,v_arr -> u',v'$ under the new basis
# <u,v> -> <u',Ginv v'>
#
# create Gram matrix for new basis "b" 
# usual convention $\vec{v} = v^i e_i = v_j e^j$ (assuming einstein sum convention)
# set of $e_i$ are the basis of $V$ and the $e^i$ are the dual basis of $V^*$
# $(e^j(e_i) = \delta^j_i $
#
# $g$ defines a natural isomorphism (map) between $V$ and $V^*$
# $G_ij = <e_i, e_j>$  for a given choice of basis
# for the standard basis of $\mathbb{R}^n$, where the bases are orthonoraml, $G_ij = \delta_{ij}$ so the components of "columns" $v^i$ are unchanged when they become "rows" via the mechanical transpose.
#
# but in our new basis, $b_i$ for i=1 to 2. is not orthonormal. Let's calculate this below. I will call the matrix `Gb`

# %%

G_diag = np.diag((newbasis_mv | newbasis_mv).e)
G_12 = (newbasis_mv[0] | newbasis_mv[1]).e
Gb = G_diag + np.array([[0.0,G_12],[G_12,0]])
Gbinv = np.linalg.inv(G)
mv.show_md("$Gb_{ij} = " f"{sp.latex(Matrix(G))}$") 
mv.show_md("$Gb^{ij} = " f"{sp.latex(Matrix(Ginv))}$")

# %%
Gb @ Gbinv

# %%

# %%
print(f"<u_mv | v_mv> = {u_mv |v_mv}")
u_arr = grade1tovector2(u_mv) # standard basis
v_arr = grade1tovector2(v_mv)
print(f"np.dot(u_arr,v_arr)={np.dot(u_arr,v_arr)}")
inner_prod_value = np.dot(u_arr,v_arr)
print(f"inner product value of <u|v> = {inner_prod_value}")
# now change to newbasis
u_arr_b, v_arr_b = Sinv @ u_arr, Sinv @ v_arr
mv.show_md(r'$ub = S^{-1} u$ to transforms compoents of $\vec{u}$ in std basis to components in new basis $b_j$')
mv.show_md(r'$ub = ' f'{sp.latex(Matrix(u_arr_b.T))}$')
mv.show_md(r'$vb = ' rf'{sp.latex(Matrix(v_arr_b))}$')
mv.show_md(r'where as the coord free versions are ' r'$\vec{u} = $' rf' ({u_mv})' r' and $\vec{v} = $' rf'({v_mv})')

# %%
newbasis, b1_mv, b2_mv, u_mv, v_mv

# %%
1.4142*b1_mv - 0.7071 * b2_mv # check that this recreates the original-yes

# %%
# alg2.graph(0x00FFE1,[v_mv,u_mv],[b1_mv, b2_mv],linewidth=3
#            ,grid=1, labels=1)

mv.show_md(rf'now show that the naive summation "dot" product does not work')
mv.show_md(rf'$<u,v>$ = {inner_prod_value} != $\sum_i ub^i vb^i$ = {einsum} = {np.dot(u_arr_b , u_arr_b)}') 

# %%
# shows that 
u_arr_b.T @ Ginv @ v_arr_b, u_arr_b.T @ Ginv @ v_arr_b == inner_prod_value

# %%
(Ginv @ u_arr_b).T @ v_arr_b == u_arr_b.T @ (Ginv @ v_arr_b)

# %%
u = np.array([1.0, 1.0])
ug  =
# %%
# %%
# so in the new basis with basis[0], basis[1] as the vasis vectors
# is ng is the gram matrix ?
# or is  g_{ij} = b_i \cdot b_j
