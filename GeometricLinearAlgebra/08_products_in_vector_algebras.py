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
# # Outline
# 1. The dot product or inner product and the dual space.
# 2. The Tensor product
# 3. The wedge product or exterior product and the exterior algebra
# 4. The geometric product
#
# ## The dot product and dual spaces
# ### vectors and vector spaces
#
# We have discussed the properties of vector spaces $V$. We keep in mind vectors in the 2D plane $\mathbb{R}^2$ and the 3D space $\mathbb{R}^3$, and throughout we write $u,v,w \in V$ for vectors and $a,b \in \mathbb{R}$ for scalars.
#
# On $\mathbb{R}^n$ the standard basis hands us a default inner product, also called the dot product: a bilinear function which takes two vectors and returns a real number. It is closely related to the projection operator, and can be used to define the projection of one vector onto a direction or line.

# %%
# draw diagrams which illustrate the geometric interpretation of the dot product
# also show the project operator as defined by the dot product

# %% [markdown]
# The linear functions $f:V \rightarrow \mathbb{R}$ themselves form a vector space, called the dual space of $V$, written $V^{*}$. Its elements obey the same vector-space axioms as the original space, and when $V$ is finite-dimensional, $\dim V^* = \dim V$. The vectors in $V^*$ are sometimes called co-vectors; other names for them are "forms" or "1-forms".
#
# Given an inner product $\langle u,v \rangle$ there is a one-to-one mapping between the elements of $V$ and $V^*$ given by:
# $$u^*(v) = \langle u,v \rangle$$
# where $u,v \in V$ and $u^* \in V^*$ is the unique linear functional which makes the identity work out.
#
# <!-- TODO(human): see the note in chat -- the sentence above and the one that
#      opens this cell now say two different things about what the inner product
#      is FOR.  One paragraph resolves it, and it is the point of section 1. -->


# %% [markdown]
# ## 2. The tensor product: where bilinearity becomes linearity
#
# The dot product takes *two* vectors and returns a number, and it is **bilinear** — linear in each
# slot separately. So is the wedge product. So is the map $(T, v) \mapsto Tv$. Bilinear maps are
# everywhere, and they are not linear maps, so none of our machinery applies to them directly.
#
# The tensor product is the construction that fixes this once and for all. Rather than study each
# bilinear map on its own terms, we build a *single new vector space* $V \otimes W$ in which the
# bilinearity has already been absorbed, so that every bilinear map on $V \times W$ becomes an
# ordinary **linear** map on $V \otimes W$.
#
# Concretely: for each pair $v \in V$, $w \in W$ we write a formal product $v \otimes w$, and we
# declare the only rules to be the ones bilinearity forces on us,
#
# $$(a v_1 + b v_2) \otimes w = a\,(v_1 \otimes w) + b\,(v_2 \otimes w), \qquad
#   v \otimes (a w_1 + b w_2) = a\,(v \otimes w_1) + b\,(v \otimes w_2).$$
#
# $V \otimes W$ is the space of all *linear combinations* of such symbols, subject to nothing else.
# That "nothing else" is the whole content of the definition, and it is what the **universal
# property** records: for every bilinear $B : V \times W \to U$ there is exactly one linear
# $\tilde B : V \otimes W \to U$ with $B(v,w) = \tilde B(v \otimes w)$. The tensor product forgets
# nothing about $B$ and adds nothing to it.
#
# Two consequences worth pinning down before we use it:
#
# - **Dimension multiplies.** If $\{b_i\}$ is a basis of $V$ and $\{c_j\}$ a basis of $W$, then
#   $\{b_i \otimes c_j\}$ is a basis of $V \otimes W$, so $\dim(V \otimes W) = \dim V \cdot \dim W$.
#   (Compare the direct sum $V \oplus W$, where dimensions *add*.)
# - **Most tensors are not simple.** An element of the form $v \otimes w$ is called a *simple* or
#   *rank-one* tensor, but these do not fill the space — they are not even a subspace. With
#   $\dim V = \dim W = 3$ the simple tensors sweep out only a $5$-dimensional cone inside a
#   $9$-dimensional space. The general element is a *sum* of simple tensors, and the fewest terms
#   needed is its **rank**.
#
# Note also that $v \otimes w \neq w \otimes v$: the tensor product is not commutative, because we
# imposed no rule saying it should be. That freedom is exactly what makes it the right raw material.
# The dot product is *symmetric* and the wedge product will be *antisymmetric*; both are obtained by
# imposing one extra rule on $V \otimes V$. Indeed
#
# $$V \otimes V \;=\; \operatorname{Sym}^2 V \;\oplus\; \Lambda^2 V, \qquad
#   n^2 = \tfrac{n(n+1)}{2} + \tfrac{n(n-1)}{2},$$
#
# which is the sense in which section 3 is a *quotient* of this section rather than a fresh start.

# %% [markdown]
# ### Linear transformations are $V \otimes V^*$
#
# Now combine the two ideas. Take $u \in V$ and a covector $\phi$ in the dual space, $\phi \in V^*$, and build the map
#
# $$L(v) = (u \otimes \phi)(v) \;=\; \phi(v)\, u .$$
#
# Read it left to right: $\phi$ **consumes** the input vector and returns a number in $\mathbb{R}$; the equation indicates that number
# scales $u$, which is **produced** as the output. This is a perfectly good linear map $V \to V$. (Check, you should see that this is a linear function. When is it a projection ($L^2=1$)?.)
#
# With a choice of coordinates it is the *outer product* of the $n$ x 1 matrix $[u]$ and the 1 x $n$ matrix $[\phi]$ — a column times a row, giving a
# matrix of rank one. For example, in $\mathbb{R}^3$, in matrix notation this would look like:
# $$
# u \otimes \phi
# \;\longleftrightarrow\;
# [u]\,[\phi]
# \;=\;
# \begin{bmatrix} u^1 \\ u^2 \\ u^3 \end{bmatrix}
# \begin{bmatrix} \phi_1 & \phi_2 & \phi_3 \end{bmatrix}
# \;=\;
# \begin{bmatrix}
# u^1\phi_1 & u^1\phi_2 & u^1\phi_3 \\
# u^2\phi_1 & u^2\phi_2 & u^2\phi_3 \\
# u^3\phi_1 & u^3\phi_2 & u^3\phi_3
# \end{bmatrix}
# $$
# and applying it to $v$ recovers the defining formula, because matrix
# multiplication is associative:
# $$
# \bigl(u\,\phi\bigr)v
# \;=\;
# u\,\bigl(\phi v\bigr)
# \;=\;
# \underbrace{\bigl(\phi_1 v^1 + \phi_2 v^2 + \phi_3 v^3\bigr)}_{\phi(v)\,\in\,\mathbb{R}}\,u .
# $$
#
#
#
# We can build all the linear maps of $V \rightarrow V$ out of these simple tensors.  Writing $\{b_i\}$ for a basis of $V$ and $\{\beta^j\}$ for the
# dual basis (the covectors that read off coordinates, $\beta^j(b_i) = \delta^j_i$), the maps
# $b_i \otimes \beta^j$ are precisely the standard basis for $V \otimes V^*$. To make this concrete, in R3, $b_1 \otimes \beta^1$ would be written in matrix formulation as:
# $$
# \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
# \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}
# \;=\;
# \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & 0 & 0 \\
# 0 & 0 & 0
# \end{bmatrix}
# $$
#
# Seeing this, I hope it makes it clear that we can make any matrix by adding up weighted combinations of the basis of simple vectors.
# $$
# L = a (b_1 \otimes \beta^1) + b (b_2 \otimes \beta^1) + c (b_3 \otimes \beta^1) + d (b_1 \otimes \beta^2) \ldots
# $$
#
# and any linear transformation of $V$, given by $L$, expands as:
#
# $$L \;=\; \sum_{i,j} L^i{}_j \; b_i \otimes \beta^j .$$
#
# So the space of linear transformations of $V$ *is* $V \otimes V^*$, and the dimension count agrees:
# $\dim V \cdot \dim V^* = n \cdot n = n^2$, the number of entries in an $n \times n$ matrix. This is
# also why index notation puts one index up and one down — the two slots are not interchangeable.
# It is $V \otimes V^*$ and **not** $V \otimes V$: one factor eats a vector, the other emits one.
#
# Two payoffs fall out immediately.
#
# **The projection operator of notebook 2 was a rank-one tensor all along.** Projection onto the
# direction $u$ is
#
# $$P_u \;=\; \frac{u \otimes u^*}{\langle u, u\rangle},$$
#
# with $u^*$ the covector attached to $u$ by the inner product, exactly as in section 1.
#
# **The trace is the canonical contraction.** There is one obvious thing to do with an element of
# $V \otimes V^*$: feed the $V$ part to the $V^*$ part. On simple tensors,
#
# $$\operatorname{tr}(u \otimes \phi) \;=\; \phi(u),$$
#
# extended linearly to all of $V \otimes V^*$. Sanity check on the identity $I = \sum_i b_i \otimes \beta^i$:
# its trace is $\sum_i \beta^i(b_i) = \sum_i 1 = n$, as it must be.
#
# "Extended linearly" hides the one step worth checking. A general $T$ is a *sum* of simple tensors,
# and that sum is not unique — so why doesn't the answer depend on which decomposition you happened
# to pick? Write $T = \sum_i u_i \otimes \phi_i$ in components, so that $T^a{}_b = \sum_i u_i^a (\phi_i)_b$,
# and compute the candidate answer:
#
# $$\sum_i \phi_i(u_i) \;=\; \sum_i \sum_a (\phi_i)_a\, u_i^a \;=\; \sum_a \Bigl(\sum_i u_i^a (\phi_i)_a\Bigr) \;=\; \sum_a T^a{}_a .$$
#
# The summation index $i$ has vanished: the right-hand side mentions only $T$ itself. So any two
# decompositions of the same tensor give the same number, and we get the coordinate formula
# $\operatorname{tr} T = T^a{}_a$ thrown in for free. (This is the universal property of section 2
# doing its job: the bilinear map $(u,\phi) \mapsto \phi(u)$ factors uniquely through $V \otimes V^*$.)
#
# More abstract note: This deserves emphasis because it is stronger than it looks. Notebook 4 defined the trace
# coordinate-free *through volume*, as $\frac{d}{dt}\big\rvert_{t=0}\det(I + tT)$. Here we get it with
# no volume form, no basis, and — crucially — **no inner product**: the pairing of $V$ against $V^*$
# is already built into the space. Contrast this with trying to contract $V \otimes V$, which has no
# canonical pairing at all; you would have to choose a metric first. The trace is free, the "trace" of a tensor in $V \otimes V$ is not.

# %%
import numpy as np

rng = np.random.default_rng(0)

# --- 1. simple tensors: tr(u ⊗ φ) = φ(u) ---------------------------------------
# A covector is stored as a plain length-3 array of components. It is a *row*; there
# is nothing to transpose, and no inner product has been used anywhere below.
u = rng.normal(size=3)
phi = rng.normal(size=3)
P = np.outer(u, phi)  # the rank-one map v ↦ φ(v)·u

print("tr(u ⊗ φ) = φ(u)      :", np.allclose(np.trace(P), phi @ u))
print("rank(u ⊗ φ) = 1       :", np.linalg.matrix_rank(P) == 1)
# the same scalar φ(u) decides idempotent vs. nilpotent:
print("P² = φ(u)·P           :", np.allclose(P @ P, (phi @ u) * P))
print(f"  here φ(u) = {phi @ u:+.4f}, so P is neither a projection nor nilpotent")


# --- 2. non-simple tensors: the trace does not care how you decompose -----------
def trace_by_contraction(us, phis):
    """Σ_i φ_i(u_i) — the definition, extended linearly off the simple tensors."""
    return sum(p @ v for v, p in zip(us, phis))


M = rng.normal(size=(3, 3))  # a general element of V ⊗ V*, rank 3, not simple

# (a) the lazy decomposition: M = Σ_a b_a ⊗ (a-th row of M)
decomp_a = (list(np.eye(3)), list(M))

# (b) the SVD: M = Σ_i σ_i u_i v_iᵀ — same tensor, completely different vectors
U, S, Vt = np.linalg.svd(M)
decomp_b = ([S[i] * U[:, i] for i in range(3)], [Vt[i, :] for i in range(3)])

# (c) a deliberately wasteful one: five *arbitrary* vectors, plus whatever covectors
#     are then forced on us to still reproduce M.  Nothing here is canonical.
W = rng.normal(size=(3, 5))
Phi = np.linalg.pinv(W) @ M  # W @ Phi == M exactly, since rank(W) = 3
decomp_c = ([W[:, i] for i in range(5)], [Phi[i, :] for i in range(5)])

for name, (us, phis) in [
    ("(a) rows,      3 terms", decomp_a),
    ("(b) SVD,       3 terms", decomp_b),
    ("(c) arbitrary, 5 terms", decomp_c),
]:
    rebuilt = sum(np.outer(v, p) for v, p in zip(us, phis))
    print(
        f"{name}:  rebuilds M = {np.allclose(rebuilt, M)}"
        f"   Σφ_i(u_i) = {trace_by_contraction(us, phis):+.6f}"
    )
print(f"{'    np.trace(M)':<22}:{'':<22}   tr M      = {np.trace(M):+.6f}")

# The identity is the canonical non-simple tensor: Σ_a b_a ⊗ β^a needs all n terms,
# because the minimum number of simple summands is exactly the rank.
print("\nrank(I) = fewest simple terms :", np.linalg.matrix_rank(np.eye(3)))
print("tr(I) = Σ_a β^a(b_a) = n      :", np.trace(np.eye(3)))


# --- 3. why it must be V ⊗ V*, and not V ⊗ V -----------------------------------
# `A` is a change of basis: its COLUMNS are the new basis vectors b'_a = Σ_c A^c_a b_c
# written in the old coordinates.  Everything below follows from that one choice:
#
#     basis vectors        transform by  A        (by construction, above)
#     vector components    transform by  A⁻¹      ("contra"variant: opposite the basis)
#     covector components  transform by  A        ("co"variant: along with the basis)
#
# So `Ainv` is not a computational convenience — it is the transformation law for an
# UPPER index, exactly as `A` is the law for a LOWER one.  Read each line below
# right-to-left against its index string; the array `M` is deliberately the same in
# both, because 9 numbers do not know what kind of tensor they are.  What separates
# V ⊗ V* from V ⊗ V is not the components, it is which law they obey.
A = rng.normal(size=(3, 3))
Ainv = np.linalg.inv(A)

# one up, one down → one A⁻¹ and one A.  This is the ordinary similarity transform of
# notebook 7: the same linear map, re-described in a new basis.
mixed = Ainv @ M @ A  # T^a_b  ↦  (A⁻¹)^a_c T^c_d A^d_b
# both up → two A⁻¹.  The `.T` is index bookkeeping, NOT a metric operation: hitting
# the second index, Σ_d T^cd (A⁻¹)^b_d, is spelled `M @ Ainv.T` as a matrix product.
contra = Ainv @ M @ Ainv.T  # T^ab   ↦  (A⁻¹)^a_c (A⁻¹)^b_d T^cd
# tr(A⁻¹MA) = tr(MAA⁻¹) = tr(M) by cyclicity; the two factors in `contra` never meet,
# so no such cancellation is available and the "trace" of a (2,0)-tensor is meaningless.

print(
    f"\ncontract up against down, T^a_a : {np.trace(M):+.6f} → {np.trace(mixed):+.6f}"
)
print(f"contract up against up,   T^aa  : {np.trace(M):+.6f} → {np.trace(contra):+.6f}")
print(
    "  trace on V ⊗ V* is basis-independent :",
    np.allclose(np.trace(M), np.trace(mixed)),
)
print(
    "  the same recipe on V ⊗ V is not      :",
    not np.allclose(np.trace(M), np.trace(contra)),
)

# Control case, and the reason anyone gets away with ignoring all of this. Restrict
# the change of basis to a ROTATION: then A⁻¹ = Aᵀ, so A⁻ᵀ = A and the two laws above
# literally coincide — upper and lower indices become indistinguishable. Euclidean
# linear algebra in orthonormal bases therefore never notices the difference between
# V ⊗ V and V ⊗ V*. It only bites once shears and rescalings are allowed back in.
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # a random orthogonal matrix
Qinv = np.linalg.inv(Q)
print("\nrestricted to an orthogonal change of basis Q:")
print("  Q⁻ᵀ = Q, so the two laws agree :", np.allclose(Qinv.T, Q))
print(
    "  both 'traces' now preserved    :",
    np.allclose(np.trace(Qinv @ M @ Q), np.trace(M))
    and np.allclose(np.trace(Qinv @ M @ Qinv.T), np.trace(M)),
)

# %% [markdown]
# ## 3. The wedge product and the exterior algebra, defined properly
#
# Notebooks 3 and 4 used $u \wedge v$ freely, justified by pictures: a signed area, an oriented plane
# element, a volume box. That was honest as *motivation* but loose as *definition* — we never said
# what kind of object $u \wedge v$ is, only what it measures. With the tensor product in hand we can
# say exactly what it is, and every property we assumed becomes a consequence rather than an
# assertion.
#
# Start from the **tensor algebra**, the direct sum of all tensor powers of $V$,
#
# $$T(V) \;=\; \bigoplus_{k \ge 0} V^{\otimes k} \;=\; \mathbb{R} \,\oplus\, V \,\oplus\, (V \otimes V) \,\oplus\, \cdots,$$
#
# with $\otimes$ itself as the multiplication. This is the free-est possible algebra built on $V$: it
# imposes no relations at all, which is precisely why section 2 could describe it as "raw material."
#
# The exterior algebra is what you get by imposing **one** relation:
#
# $$\boxed{\;\Lambda V \;=\; T(V) \,\big/\, \big\langle\, v \otimes v \;:\; v \in V \,\big\rangle\;}$$
#
# — the quotient of $T(V)$ by the two-sided ideal generated by all squares. We write the induced
# product as $\wedge$. That single rule, $v \wedge v = 0$, generates everything.
#
# ### Antisymmetry is a theorem, not an axiom
#
# It is worth doing this one derivation explicitly, because the two properties are usually stated as
# if they were interchangeable. Apply the rule to the sum $u + v$:
#
# $$0 = (u+v) \wedge (u+v) = \underbrace{u \wedge u}_{0} + u \wedge v + v \wedge u + \underbrace{v \wedge v}_{0}
#   \quad\Longrightarrow\quad u \wedge v = -\,v \wedge u .$$
#
# So **alternating implies antisymmetric**. The converse needs division by $2$, so over $\mathbb{R}$
# the two are equivalent — but $v \wedge v = 0$ is the stronger, more primitive statement, and it is
# the right axiom to take. It also says directly what the geometry demanded all along: a
# parallelogram with two equal edges has no area.
#
# ### As a subspace instead of a quotient
#
# Over $\mathbb{R}$ we may equally realise $\Lambda^2 V$ *inside* $V \otimes V$ as the antisymmetric
# tensors, via
#
# $$u \wedge v \;=\; u \otimes v - v \otimes u .$$
#
# Be warned that this is a genuine convention: many texts insert a factor of $\tfrac{1}{2}$ (or
# $\tfrac{1}{k!}$ in degree $k$) so that the identification with the quotient is an isometry rather
# than merely an isomorphism. Nothing conceptual hangs on it, but determinant and norm formulas pick
# up powers of $k!$ depending on the choice, which is one of the places "loose" definitions actually
# bite. We use the factor-free convention above.
#
# This is also where section 2's warning about simple tensors returns. Recall that $V \otimes V$
# splits as $\operatorname{Sym}^2 V \oplus \Lambda^2 V$; the wedge is exactly the projection onto the
# second summand, and the dot product — symmetric — lives in the first. **The two products of this
# series are the two halves of the tensor product.**
#
# ### Higher degrees and the shape of $\Lambda V$
#
# The same quotient in degree $k$ kills any product with a repeated factor, leaving
# $\dim \Lambda^k V = \binom{n}{k}$, with basis $\{b_{i_1} \wedge \cdots \wedge b_{i_k}\}$ over
# strictly increasing index tuples. Summing over $k$ gives $\dim \Lambda V = 2^n$: the exterior
# algebra of an $n$-dimensional space has one basis element per *subset* of a basis of $V$. For
# $V = \mathbb{R}^3$:
#
# | degree | $\Lambda^0$ | $\Lambda^1$ | $\Lambda^2$ | $\Lambda^3$ |
# |---|---|---|---|---|
# | dimension | 1 | 3 | 3 | 1 |
# | name | scalars | vectors | bivectors | pseudoscalars |
#
# The product is associative and **graded-commutative**: for $\alpha \in \Lambda^k$ and
# $\beta \in \Lambda^l$, $\;\alpha \wedge \beta = (-1)^{kl}\, \beta \wedge \alpha$. Vectors
# anticommute, but bivectors commute with each other.
#
# Three things this immediately explains, all of which we previously took on faith:
#
# - **The determinant.** $\Lambda^n V$ is one-dimensional, so any linear $T$ acts on it by a single
#   number. That number is $\det T$ — which is what notebook 4 defined by volume, now recovered as a
#   dimension count rather than a geometric appeal.
# - **Why the cross product is a three-dimensional accident.** $\dim \Lambda^2 \mathbb{R}^3 = 3$,
#   equal to $\dim \mathbb{R}^3$, so in 3D — and, checking $\binom{n}{2} = n$, in *no other
#   dimension* — a bivector can be disguised as a vector. That disguise is the cross product, and it
#   is why $u \times v$ fails to generalise while $u \wedge v$ does.
# - **Bivectors are not all planes.** In $\mathbb{R}^3$ every bivector happens to be simple, i.e.
#   expressible as a single $u \wedge v$, so the picture "a bivector is an oriented plane patch" is
#   safe there. It fails as soon as $n \ge 4$: in $\mathbb{R}^4$ the bivector
#   $e_1 \wedge e_2 + e_3 \wedge e_4$ is not simple, and represents no plane at all. This is the
#   same non-simplicity we met in section 2, and it is why "bivector" and "plane element" must be
#   kept distinct.

# %%
# TODO(human): numerical checks for section 3 -- see the note in chat.

# %% [markdown]
#
