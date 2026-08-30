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
# # Geometric Linear Algebra 9 — Matrices from a Basis: Columns for $V$, Rows for $V^*$
#
# Notebook 8 built the dual space $V^*$ abstractly: the linear functions $V \to \mathbb{R}$, a vector
# space in its own right, related to $V$ only *after* you choose an inner product. Notebook 5 showed
# that choosing a basis turns arrows into coordinates. This notebook combines the two and asks the
# concrete question: **once a basis is fixed, what do $V$ and $V^*$ look like as arrays of numbers?**
#
# The answer is the oldest convention in linear algebra, and it turns out to be forced rather than
# arbitrary. Vectors become **column** vectors, covectors become **row** vectors, and the pairing
# $\phi(v)$ becomes ordinary matrix multiplication — a $1 \times n$ times an $n \times 1$, giving the
# $1 \times 1$ scalar. No transpose, no dot product, no metric is needed to make that work.
#
# Three payoffs follow, each of which explains something usually presented as a rule to memorise:
#
# - **Why components transform two different ways.** Change the basis and vector components transform
#   by $P^{-1}$ while covector components transform by $P$. These are inverse to one another, which is
#   *precisely* why the number $\phi(v)$ is unchanged. Contravariant and covariant are not two
#   arbitrary conventions; they are the two halves of one cancellation.
# - **Why the transpose exists without a metric.** The dual map $T^*$ acting on covectors is just
#   right-multiplication by $[T]$. Writing covectors as columns instead turns that into $[T]^{\mathsf T}$.
#   The transpose is a statement about $V^*$, not about lengths.
# - **Why none of this was ever visible before.** In an orthonormal basis the Gram matrix is the
#   identity, index lowering does nothing, and a row vector really is just the transpose of a column.
#   The whole $V$ versus $V^*$ distinction collapses — which is why it can be ignored for years and
#   then suddenly matters.
#
# We work in the plane so everything can be drawn, and verify each claim numerically in a
# deliberately **non-orthonormal** basis, since an orthonormal one would make every identity true for
# the wrong reason.

# %%
import numpy as np
import clmmathtools.viz as mv  # shared plane-viz library (see ../clmmathtools)

BLUE = 0x4FC3F7
ORANGE = 0xFFB74D
GREEN = 0x81C784
RED = 0xEF5350
PURPLE = 0xCE93D8
GREY = 0x777777
FAINT = 0x3A3A4E

BG = 0x0F0F0F
GRIDC = 0x333333
LABELC = 0xCCCCCC

np.set_printoptions(precision=4, suppress=True)


# %% [markdown]
# ### Toolkit (same helpers as the rest of the series)


# %%
def new_plot(lim=3.0, axes=True, grid=True):
    return mv.Plane(extent=lim, grid=grid, axes=axes)


def add_vector(plot, tail, head, color, label=None):
    tail = np.asarray(tail, float).reshape(-1)
    head = np.asarray(head, float).reshape(-1)
    plot.vector(head - tail, origin=tail, color=color, label=label)
    return plot


def add_line(plot, pts, color, width=0.02, alpha=1.0):
    plot.curve(pts, color=color, width=max(1.0, width * 130), alpha=alpha)
    return plot


def check(label, got, want, tol=1e-10):
    """Print a numeric identity and assert it, in the style of the rest of the series."""
    ok = np.allclose(got, want, atol=tol)
    print(f"  {'OK ' if ok else 'FAIL'}  {label}")
    assert ok, f"{label}: {got} != {want}"


# %% [markdown]
# ## 1. A basis, and the dual basis it drags along
#
# Fix a basis $\{b_1, b_2\}$ of $V$ and collect the basis vectors as the **columns** of a matrix $B$.
# (Notebook 5 stored them as rows; here columns are worth the switch, because the point of this
# notebook is that the column/row split carries meaning.)
#
# A basis of $V$ automatically determines a basis of $V^*$ — the **dual basis** $\{\beta^1, \beta^2\}$,
# defined by the requirement that each $\beta^i$ reads off the $i$-th coordinate:
#
# $$\beta^i(b_j) = \delta^i_j .$$
#
# Stack the $\beta^i$ as the **rows** of a matrix and that requirement says exactly $\beta B = I$. So
# the dual basis is $B^{-1}$, read by rows. Note this is a completely metric-free construction: no
# angle, no length, no dot product went into it — only the ability to solve linear equations.

# %%
# columns of B are the basis vectors b_1, b_2  (a skewed, non-orthonormal frame)
B = np.array([[1.0, 0.6], [0.0, 1.0]])
Beta = np.linalg.inv(B)  # ROWS are the dual covectors beta^1, beta^2

print("b_1 =", B[:, 0], "   b_2 =", B[:, 1])
print("beta^1 =", Beta[0], "   beta^2 =", Beta[1])
print("\ndefining property beta^i(b_j) = delta^i_j:")
for i in range(2):
    for j in range(2):
        print(f"  beta^{i + 1}(b_{j + 1}) = {Beta[i] @ B[:, j]:.4f}")
check("beta B = I", Beta @ B, np.eye(2))

# The dual basis is NOT the original basis in disguise -- beta^1 is not parallel to b_1.
cos = (B[:, 0] @ Beta[0]) / (np.linalg.norm(B[:, 0]) * np.linalg.norm(Beta[0]))
print(
    f"\nangle between b_1 and beta^1: cos = {cos:.4f}  (would be 1.0 if they coincided)"
)
print(
    "beta^1 is orthogonal to b_2, not aligned with b_1:",
    f"beta^1 . b_2 = {Beta[0] @ B[:, 1]:.4f}",
)

# %% [markdown]
# That last line is the geometric content of the dual basis, and it is worth staring at:
# $\beta^1$ is the covector that **annihilates every other basis vector**. It points "across" $b_2$,
# not "along" $b_1$. In an orthonormal frame those two descriptions coincide, which is the source of
# the usual confusion.

# %%
p = new_plot(lim=2.2)
add_vector(p, [0, 0], B[:, 0], BLUE, label=r"$b_1$")
add_vector(p, [0, 0], B[:, 1], ORANGE, label=r"$b_2$")
add_vector(p, [0, 0], Beta[0], GREEN, label=r"$\beta^1$")
add_vector(p, [0, 0], Beta[1], PURPLE, label=r"$\beta^2$")
p.display()

# %% [markdown]
# ## 2. The pairing is row times column — no metric required
#
# A vector $x \in V$ has **column** components $v$ in the basis $B$, meaning $x = Bv$, so
# $v = B^{-1}x$. A covector $\phi \in V^*$ has **row** components $\phi_B$ in the dual basis. The
# pairing is then literally matrix multiplication:
#
# $$\phi(x) \;=\; \underbrace{\phi_B}_{1 \times n}\;\underbrace{v}_{n \times 1}.$$
#
# The shapes only fit one way round, and that is the whole justification for the convention. Contrast
# this with the dot product of two *vectors*, $u \cdot v$, which as matrices is $u^{\mathsf T} v$ — a
# transpose had to be inserted by hand, because both arguments were columns. **The transpose in
# $u^{\mathsf T}v$ is doing the job of the metric**, silently converting a vector into a covector. In
# the pairing above nothing has to be converted, so nothing has to be chosen.

# %%
x = np.array([1.8, 1.2])  # a geometric vector, in standard coordinates
phi_std = np.array([2.0, 1.0])  # a geometric covector, as a standard-coordinate row

v = Beta @ x  # column components of x in basis B
phi_B = phi_std @ B  # row components of phi in the dual basis

print("x  (standard coords)     =", x)
print("v  (components in B)     =", v, "   -> x = B v :", B @ v)
print("phi (standard row)       =", phi_std)
print("phi_B (dual-basis row)   =", phi_B)
print()
print(f"phi(x) computed in standard coords : {phi_std @ x: .8f}")
print(f"phi(x) computed in the B basis     : {phi_B @ v: .8f}")
check("pairing is basis-independent", phi_B @ v, phi_std @ x)

# %% [markdown]
# ## 3. Operators: the matrix is a record of where the basis went
#
# A linear map $T$ becomes a matrix by the rule "column $j$ holds the coordinates of $T b_j$". In the
# basis $B$ that is the similarity $[T]_B = B^{-1} T B$, which is just "convert in, act, convert out".

# %%
T = np.array([[1.0, -0.8], [0.5, 1.2]])
T_B = Beta @ T @ B

print("[T] in standard coords:\n", T)
print("[T] in basis B:\n", T_B)
print("\ncolumn j of [T]_B should be the B-coordinates of T b_j:")
for j in range(2):
    print(
        f"  T b_{j + 1} = {T @ B[:, j]}  ->  coords {Beta @ (T @ B[:, j])}  vs column {T_B[:, j]}"
    )
    check(f"column {j + 1}", T_B[:, j], Beta @ (T @ B[:, j]))

print()
check("trace is basis-independent", np.trace(T_B), np.trace(T))
check("det is basis-independent", np.linalg.det(T_B), np.linalg.det(T))
print(f"  trace = {np.trace(T):.6f},  det = {np.linalg.det(T):.6f}")

# %% [markdown]
# ## 4. The two transformation laws, and why the pairing survives them
#
# Now change basis. Let the new basis be $b'_j = \sum_i P^i{}_j\, b_i$, i.e. $B' = BP$. Chasing the
# definitions gives the two laws that index notation encodes with raised and lowered indices:
#
# $$v' = P^{-1} v \qquad\text{(vector components, }\textit{contra}\text{variant)}$$
# $$\phi' = \phi\, P \qquad\text{(covector components, }\textit{co}\text{variant)}$$
#
# Vector components transform *oppositely* to the basis — stretch the basis vectors and the numbers
# describing a fixed arrow shrink. Covector components transform *with* it. Multiply the two laws
# together and the $P$'s annihilate:
#
# $$\phi' v' = (\phi P)(P^{-1} v) = \phi v .$$
#
# That cancellation is not a lucky accident; it is the reason the two laws must be inverse to each
# other. A quantity is *geometric* exactly when its coordinate description transforms so as to leave
# the observable pairings alone.

# %%
P = np.array([[1.0, 0.5], [0.3, 1.0]])
Bp = B @ P  # the new basis
vp = np.linalg.inv(Bp) @ x  # components of the SAME x in the new basis
phi_Bp = phi_std @ Bp  # row components of the SAME phi in the new dual basis

print("same arrow x, three sets of numbers:")
print("  standard :", x)
print("  in B     :", v)
print("  in B'    :", vp)
check("v' = P^-1 v  (contravariant)", vp, np.linalg.inv(P) @ v)

print("\nsame covector phi, three sets of numbers:")
print("  standard :", phi_std)
print("  in B     :", phi_B)
print("  in B'    :", phi_Bp)
check("phi' = phi P  (covariant)", phi_Bp, phi_B @ P)

print(f"\npairing in B  : {phi_B @ v: .8f}")
print(f"pairing in B' : {phi_Bp @ vp: .8f}")
check("the P's cancel", phi_Bp @ vp, phi_B @ v)

# %% [markdown]
# ## 5. The dual map is the transpose — and no inner product appeared
#
# Given $T : V \to V$ there is a map going the *other way* on covectors, the **dual** (or pullback)
# map $T^* : V^* \to V^*$, defined by composition:
#
# $$(T^*\phi)(v) \;=\; \phi(Tv).$$
#
# With $\phi$ a row and $v$ a column this is $(\phi T) v$ — so $T^*$ is simply **right-multiplication
# by $[T]$**. If instead you insist on writing covectors as columns, right-multiplication becomes
# left-multiplication by $[T]^{\mathsf T}$, and the transpose appears.
#
# This is worth separating from the usual story. The transpose here required no lengths, no angles
# and no orthonormality — it is bookkeeping for a map between dual spaces, and it reverses direction.
# The *other* transpose, the adjoint $T^\dagger : V \to V$ with
# $\langle T^\dagger u, v\rangle = \langle u, Tv\rangle$, genuinely does need an inner product. They
# coincide in an orthonormal basis, which is why one symbol gets used for both.

# %%
y = np.array([0.7, -1.4])
print(f"(T* phi)(y) = {(phi_std @ T) @ y: .8f}")
print(f"phi(T y)    = {phi_std @ (T @ y): .8f}")
check("(T* phi)(y) = phi(T y)", (phi_std @ T) @ y, phi_std @ (T @ y))
check("as a column-convention matrix, T* is T^T", (phi_std @ T), T.T @ phi_std)
print("\n  -- no inner product was used anywhere in this cell --")

# %% [markdown]
# ## 6. The Gram matrix, and why none of this was ever visible
#
# Everything so far avoided the metric. Reintroduce it and we can finally connect a vector to a
# covector, as in notebook 8's $u^*(v) = \langle u, v\rangle$. In coordinates that map is the **Gram
# matrix**
#
# $$g_{ij} = \langle b_i, b_j \rangle, \qquad\text{i.e.}\qquad g = B^{\mathsf T} B,$$
#
# and converting a vector's components into the corresponding covector's components is
# $\phi_i = g_{ij} v^j$ — the operation physicists call **lowering an index**.
#
# Now the punchline. If the basis is orthonormal then $g = I$, lowering does nothing at all, and the
# row components of $u^*$ are numerically identical to the column components of $u$. A row vector
# becomes indistinguishable from "a column vector, transposed", and $V$ and $V^*$ appear to be the
# same space. They are not — the identification was smuggled in by the choice of basis, which is
# exactly the non-canonical isomorphism notebook 8 warned about.

# %%
g = B.T @ B
print("Gram matrix g = B^T B:\n", g)
print("g symmetric:", np.allclose(g, g.T), "| g == I:", np.allclose(g, np.eye(2)))

lowered = v @ g  # row components of the metric-dual of x, in the dual basis
w = np.array([0.4, 1.1])
w_comp = Beta @ w
print(f"\nlowered(x) applied to w : {lowered @ w_comp: .8f}")
print(f"<x, w> directly          : {x @ w: .8f}")
check("lowering reproduces the inner product", lowered @ w_comp, x @ w)

# Now redo it in an ORTHONORMAL basis and watch the distinction evaporate.
theta = 0.7
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
g_orth = Q.T @ Q
x_q = np.linalg.inv(Q) @ x
print("\northonormal basis Q:  g =\n", g_orth)
check("g = I for an orthonormal basis", g_orth, np.eye(2))
check("lowering is now the identity on components", x_q @ g_orth, x_q)
print("  -> row = column^T, and the V / V* distinction becomes invisible")

# %% [markdown]
# ## 7. Drawing the difference
#
# The algebra above says vectors and covectors transform oppositely, so they should not *look* alike
# either. The standard picture, due to Misner–Thorne–Wheeler, is that a covector is not an arrow but
# a **stack of parallel level lines** — the sets $\phi = \ldots, -1, 0, 1, 2, \ldots$ — and that
# $\phi(v)$ counts how many of those lines the arrow $v$ pierces.
#
# Two facts make the picture faithful, both easy to verify:
#
# - the level lines of $\phi$ are spaced $1/\lVert\phi\rVert$ apart, so a **larger covector is a
#   denser stack**, not a longer arrow;
# - scaling the basis up makes vector components shrink and covector components grow — the arrow
#   looks shorter while the stack gets denser.

# %%
phi2 = np.array([2.0, 1.0])
spacing = 1.0 / np.linalg.norm(phi2)
print(f"covector {phi2}: level lines spaced {spacing:.6f} apart = 1/|phi| = 1/sqrt(5)")
check("spacing = 1/|phi|", spacing, 1 / np.sqrt(5))

print("\nphi(v) counts pierced level lines:")
for vec in ([1.0, 0.0], [0.0, 1.0], [1.5, 1.0], [2.0, 2.0]):
    vec = np.array(vec)
    print(
        f"  phi . {vec} = {phi2 @ vec:5.2f}  -> pierces {abs(phi2 @ vec):.2f} unit lines"
    )

# %%
# TODO(human): draw the covector `phi2` as a stack of level lines, together with a few
# vectors, so that the pairing becomes visibly a count of crossings.
#
# `mv.Plane.line(point, direction, color=..., width=..., alpha=...)` draws an infinite line;
# the level set phi = c is the line through (c/|phi|^2) * phi with direction perpendicular
# to phi.  See the note in chat for the design decision this cell is really about.
#
# p = new_plot(lim=2.5)
# ...
# p.display()

# %% [markdown]
# ## Summary
#
# Choosing a basis of $V$ forces a dual basis of $V^*$ (the rows of $B^{-1}$), and with it the entire
# column/row convention of matrix algebra:
#
# - **vectors are columns, covectors are rows**, and the pairing $\phi(v)$ is matrix multiplication —
#   the shapes fit only one way, and no metric is consulted;
# - **components transform inversely** ($v' = P^{-1}v$ against $\phi' = \phi P$), which is not a pair
#   of conventions to memorise but a single cancellation that keeps $\phi(v)$ observable;
# - **an operator's matrix records where the basis went**, and changing basis conjugates it,
#   $[T]_{B'} = P^{-1}[T]_B P$, leaving trace and determinant alone — the belated justification for
#   notebook 4 defining them without coordinates;
# - **the dual map $T^*$ is the transpose**, arrived at with no inner product anywhere; the metric
#   adjoint $T^\dagger$ is a different map that happens to agree in an orthonormal basis;
# - **the Gram matrix $g = B^{\mathsf T}B$ is the only thing that ever connects $V$ to $V^*$**, and it
#   degenerates to $I$ exactly when the basis is orthonormal — which is why a first course in linear
#   algebra can treat rows as transposed columns and never notice a thing.
#
# Notebook 8 argued abstractly that $V \cong V^*$ is not canonical. This notebook is what that claim
# looks like when you actually compute: the isomorphism is a matrix, it is called $g$, and it is the
# identity only because someone chose an orthonormal basis.

# %% [markdown]
#
