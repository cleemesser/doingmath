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
# # Low-Rank Functions and Sketching
#
# ### Approximating information-dense functions by sums of simple pieces — and finding those pieces cheaply
#
# The idea in one line: a complicated function of several variables can often be written, to high
# accuracy, as a **short sum of products of one-variable functions**,
#
# $$f(x, y) \;\approx\; \sum_{k=1}^{r} u_k(x)\, v_k(y), \qquad
#   f(x, y, z) \;\approx\; \sum_{k=1}^{r} u_k(x)\, v_k(y)\, w_k(z),$$
#
# and when it can, the function is far cheaper to store, evaluate, integrate and learn than its raw
# samples suggest. Sampled on a grid, the left formula is a **low-rank matrix factorization** and the
# right one is a **CP (canonical polyadic) tensor decomposition**. The number of terms $r$ is the
# *rank*, and it is a measure of how much genuinely joint structure the function has.
#
# **A note on the word "sketching".** It is used for two related things, and it helps to keep them
# apart. The *representation* — a low-rank matrix, a CP or Tucker or tensor-train format — is the
# compressed object. **Sketching** in the narrow, numerical-linear-algebra sense is the *method*:
# multiply the big object by a small **random** matrix, and recover a low-rank approximation from the
# much smaller product, without ever computing a full SVD and often without ever looking at the whole
# object. This notebook covers both, in that order: first *why* functions are low rank, then *how* to
# find the low-rank form cheaply by randomization or by sampling a few rows and columns.
#
# As elsewhere in this repository, every claim is checked in code.
#
# **Contents**
#
# 1. A function of two variables is a matrix; rank counts separable terms
# 2. Why random projections work at all: Johnson–Lindenstrauss
# 3. The randomized range finder (Halko–Martinsson–Tropp), and where it fails
# 4. A single-pass sketch: compressing what you can only see once
# 5. Cross approximation: building the low-rank form from a few *function evaluations*
# 6. Three variables: CP and Tucker, and the surprising ways tensor rank misbehaves
# 7. Many variables: tensor trains and the curse of dimensionality
# 8. Where this shows up: EEG, network compression, latent spaces
# * References

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker

warnings.filterwarnings(
    "ignore", category=SyntaxWarning
)  # tensorly docstrings on Python 3.14
tl.set_backend("numpy")
rng = np.random.default_rng(0)

plt.rcParams.update(
    {
        "figure.facecolor": "#0f0f0f",
        "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#ccc",
        "text.color": "#eee",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "monospace",
    }
)

BLUE = "#4fc3f7"
ORANGE = "#ffb74d"
GREEN = "#81c784"
RED = "#ef5350"
PURPLE = "#ce93d8"
YELLOW = "#ffd54f"
GREY = "#888888"

# %% [markdown]
# ## 1. A function of two variables is a matrix; rank counts separable terms
#
# Sample $f(x,y)$ on an $n\times n$ grid and you have a matrix $A_{ij} = f(x_i, y_j)$. A **separable**
# function $f(x,y) = u(x)v(y)$ gives a rank-one matrix $\mathbf u\mathbf v^{\mathsf T}$, so the rank of
# $A$ is the smallest number of separable terms that reproduce $f$ on the grid.
#
# The singular value decomposition $A = \sum_k \sigma_k \mathbf u_k \mathbf v_k^{\mathsf T}$ then
# answers the approximation question completely. **Eckart and Young (1936)** proved that truncating
# it after $r$ terms gives the best rank-$r$ approximation in both the spectral and Frobenius norms,
#
# $$\min_{\operatorname{rank}B\le r}\lVert A - B\rVert_2 = \sigma_{r+1}, \qquad
#   \min_{\operatorname{rank}B\le r}\lVert A - B\rVert_F = \Big(\sum_{k>r}\sigma_k^2\Big)^{1/2},$$
#
# so **how fast the singular values decay is how compressible the function is**. The useful
# summary is the $\varepsilon$-rank: the number of $\sigma_k$ above $\varepsilon\,\sigma_1$.

# %%
n = 400
x = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, x, indexing="ij")

examples = {
    "1/(1+x+y)": 1.0 / (1.0 + X + Y),
    "exp(-(x-y)²/0.01)": np.exp(-((X - Y) ** 2) / 0.01),
    "|x - y|": np.abs(X - Y),
    "1[x < y]": (X < Y).astype(float),
    "1[x < 0.5]": (X < 0.5).astype(float),
}


def eps_rank(A, eps):
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > eps * s[0]))


print(f"{'f(x, y)':>20}   ε-rank at 1e-3   1e-6   1e-12     (grid is {n}×{n})")
for name, A in examples.items():
    print(
        f"{name:>20}   {eps_rank(A, 1e-3):>12d}{eps_rank(A, 1e-6):>7d}{eps_rank(A, 1e-12):>8d}"
    )

# Eckart–Young, checked: the truncated SVD's error is exactly the next singular value
A = examples["exp(-(x-y)²/0.01)"]
U, s, Vt = np.linalg.svd(A)
for r in (5, 15, 25):
    Ar = (U[:, :r] * s[:r]) @ Vt[:r]
    print(
        f"rank {r:2d}: ‖A − A_r‖₂ = {np.linalg.norm(A - Ar, 2):.3e}   σ_(r+1) = {s[r]:.3e}"
        f"   equal: {np.isclose(np.linalg.norm(A - Ar, 2), s[r])}"
    )

# %% [markdown]
# Read the table carefully; it says three different things.
#
# * **Smooth functions are low rank.** $1/(1+x+y)$ needs 7 terms for twelve digits on a 400×400 grid.
#   This is not luck: a function analytic in a neighbourhood of the domain has singular values that
#   decay geometrically, because it can be expanded in a rapidly converging separable series. (For
#   this Cauchy-like kernel the decay is governed by a rational-approximation problem — Beckermann &
#   Townsend 2017.)
# * **Narrow features cost rank.** The Gaussian ridge is just as smooth, but its width $0.1$ means it
#   takes ~40 terms: the rank tracks how many "wiggles" one-variable functions need to resolve it.
# * **Rank is about alignment with the axes, not about smoothness.** The last two rows are both
#   discontinuous indicator functions. $\mathbf 1[x<y]$ — a step along the *diagonal* — is full rank;
#   $\mathbf 1[x<0.5]$ — a step along a *coordinate* — is rank one, because it does not depend on $y$
#   at all. Low rank means "nearly separable in *these* coordinates." Rotate the coordinates and the
#   rank can change completely. That is the first thing to remember when choosing a representation.

# %%
fig = plt.figure(figsize=(16.5, 5.4))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
fig.suptitle(
    "Compressibility = singular-value decay. Smooth, axis-aligned structure is cheap; diagonal structure is not.",
    color="white",
    fontsize=11,
)
colors = [BLUE, GREEN, ORANGE, RED, PURPLE]
for k, ((name, Ak), c) in enumerate(zip(examples.items(), colors)):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(Ak, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    ax.set_title(name, color=c, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
axs = fig.add_subplot(gs[1, :])
for (name, Ak), c in zip(examples.items(), colors):
    sk = np.linalg.svd(Ak, compute_uv=False)
    axs.semilogy(
        np.arange(1, 81),
        np.maximum(sk[:80] / sk[0], 1e-17),
        "-",
        color=c,
        lw=2,
        label=name,
    )
axs.axhline(1e-12, color=GREY, lw=1, ls=":")
axs.set_xlabel("index k")
axs.set_ylabel("σ_k / σ_1")
axs.set_ylim(1e-17, 2)
axs.grid(True)
axs.legend(fontsize=8, ncol=5, loc="upper right")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# ## 2. Why random projections work at all: Johnson–Lindenstrauss
#
# Sketching rests on a geometric fact that sounds too good to be true. **Johnson and Lindenstrauss
# (1984)** showed that any $N$ points in $\mathbb{R}^D$ can be mapped into $\mathbb{R}^k$ with
#
# $$k = O\!\left(\varepsilon^{-2}\log N\right)$$
#
# while every pairwise distance is preserved to within a factor $1\pm\varepsilon$ — and that a
# *random* Gaussian matrix does it with high probability. Note what $k$ does **not** depend on: the
# ambient dimension $D$. The projection need only be large enough to tell $N$ points apart.
#
# For low-rank approximation the relevant version is a subspace one: a random $k$-dimensional
# sample of the range of $A$ captures its dominant $r$-dimensional part once $k$ exceeds $r$ by a few
# — because a random direction is very unlikely to be nearly orthogonal to a fixed subspace.

# %%
N, D = 200, 10_000
P = rng.normal(size=(N, D))
iu, ju = np.triu_indices(N, 1)
d_orig = np.linalg.norm(P[iu] - P[ju], axis=1)

print(
    f"{N} points in ℝ^{D}; all {len(iu):,} pairwise distances, before and after a random projection\n"
)
print(f"{'k':>6}{'min ratio':>12}{'max ratio':>12}{'spread·√k':>12}")
for k in (25, 100, 400, 1600):
    R = rng.normal(size=(D, k)) / np.sqrt(k)
    Q = P @ R
    ratio = np.linalg.norm(Q[iu] - Q[ju], axis=1) / d_orig
    print(
        f"{k:>6}{ratio.min():>12.3f}{ratio.max():>12.3f}{(ratio.max() - ratio.min()) * np.sqrt(k):>12.2f}"
    )
print(
    "\nThe distortion shrinks like 1/√k (the last column is roughly constant) and D never appears:"
)
print(
    "10,000 dimensions are squeezed to 400 with every one of 19,900 distances kept within ~15%."
)

# %% [markdown]
# ## 3. The randomized range finder, and where it fails
#
# **Halko, Martinsson and Tropp (2011)** turned the JL intuition into the standard algorithm. To find
# a rank-$k$ approximation of an $m\times n$ matrix $A$:
#
# 1. Draw a Gaussian test matrix $\Omega \in \mathbb{R}^{n\times(k+p)}$, with a small oversampling
#    $p \approx 5$–$10$.
# 2. Form the **sketch** $Y = A\Omega$ — $k+p$ random combinations of the columns of $A$.
# 3. Orthonormalize: $Y = QR$. The columns of $Q$ approximately span the dominant range of $A$.
# 4. Project: $B = Q^{\mathsf T}A$, a small $(k+p)\times n$ matrix, and take *its* SVD.
#
# The only access to $A$ is two products, $A\Omega$ and $Q^{\mathsf T}A$, so $A$ never has to be
# formed — it can be an operator, a kernel evaluated on the fly, or a matrix too large for memory.
# Their main theorem bounds the expected error by $\sigma_{k+1}$ times a modest factor that also
# involves the *tail* $\big(\sum_{j>k}\sigma_j^2\big)^{1/2}$: the method is near-optimal when the
# spectrum decays fast, and needs help when it does not.
#
# The help is **power iteration**: sketch $(AA^{\mathsf T})^qA$ instead of $A$, which raises every
# singular value to the power $2q+1$ and sharpens the decay. There is a trap, which we will spring
# deliberately.


# %%
def rsvd(A, k, p=10, q=0, stable=True, seed=0):
    """Randomized SVD (Halko–Martinsson–Tropp 2011, Alg. 4.3/4.4 + 5.1), returning the rank-k matrix.

    stable=False applies (AAᵀ)^q directly — the version that looks right and is not.
    """
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(A @ r.normal(size=(A.shape[1], k + p)))
    for _ in range(q):
        if stable:  # re-orthonormalize after every application (their Alg. 4.4)
            W, _ = np.linalg.qr(A.T @ Q)
            Q, _ = np.linalg.qr(A @ W)
        else:
            Q = A @ (A.T @ Q)
    if not stable and q:
        Q, _ = np.linalg.qr(Q)
    Ub, sb, Vbt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return (Q @ Ub[:, :k]) * sb[:k] @ Vbt[:k]


A_fast = examples["exp(-(x-y)²/0.01)"]  # fast spectral decay
A_flat = np.abs(X - Y) + 1e-2 * rng.normal(size=(n, n)) / np.sqrt(
    n
)  # slow decay + noise floor
s_fast = np.linalg.svd(A_fast, compute_uv=False)
s_flat = np.linalg.svd(A_flat, compute_uv=False)

print(
    "error of the randomized SVD divided by the optimal error σ_(k+1)   (1.00 = optimal)\n"
)
for label, Am, sm in [
    ("fast decay (Gaussian ridge)", A_fast, s_fast),
    ("slow decay (|x−y| + noise)", A_flat, s_flat),
]:
    print(label)
    for k in (10, 25, 40):
        cells = [
            f"q={q}: {np.linalg.norm(Am - rsvd(Am, k, q=q), 2) / sm[k]:5.2f}"
            for q in (0, 1, 2)
        ]
        print(f"   k = {k:2d}:   " + "   ".join(cells))

print(
    "\nThe trap — power iteration without re-orthonormalization, fast-decay matrix, k = 30, q = 2:"
)
for stable in (False, True):
    ratio = (
        np.linalg.norm(A_fast - rsvd(A_fast, 30, q=2, stable=stable), 2) / s_fast[30]
    )
    print(f"   stable = {stable!s:5}:  error / optimal = {ratio:8.2f}")

# %% [markdown]
# Three lessons, in order of how often they bite.
#
# * **With fast decay, one sketch is already optimal.** For the Gaussian ridge every ratio is $1.00$:
#   the random sample finds the dominant subspace on the first try.
# * **With slow decay, power iteration earns its keep.** For $\lvert x-y\rvert$ plus noise, $q=0$ at
#   $k=40$ is ~2× worse than optimal; one power iteration fixes it.
# * **Done naively, power iteration destroys accuracy.** Multiplying by $AA^{\mathsf T}$ without
#   re-orthonormalizing squares the dynamic range each time, and in floating point everything below
#   about $\sigma_1\sqrt{\varepsilon_{\text{mach}}}$ is rounded away. The fast-decay matrix — the case
#   that needed no help — is made ~300× *worse*. Halko, Martinsson and Tropp flag exactly this (their
#   §4.5, Algorithm 4.4); it is the most common error in hand-rolled implementations.

# %%
ks = np.arange(2, 61, 2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Randomized SVD error vs the Eckart–Young optimum σ_(k+1)",
    color="white",
    fontsize=11,
)
for ax, (title, Am, sm) in zip(
    axes,
    [
        ("fast decay: Gaussian ridge", A_fast, s_fast),
        ("slow decay: |x − y| + noise", A_flat, s_flat),
    ],
):
    ax.semilogy(ks, sm[ks], "-", color=GREY, lw=3, label="optimal σ_(k+1)")
    for q, c in [(0, BLUE), (1, GREEN)]:
        ax.semilogy(
            ks,
            [np.linalg.norm(Am - rsvd(Am, k, q=q), 2) for k in ks],
            "o",
            color=c,
            ms=4,
            label=f"randomized, q = {q}",
        )
    if Am is A_fast:
        ax.semilogy(
            ks,
            [np.linalg.norm(Am - rsvd(Am, k, q=2, stable=False), 2) for k in ks],
            "x",
            color=RED,
            ms=6,
            label="q = 2, no re-orthonormalization",
        )
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_xlabel("target rank k")
    ax.set_ylabel("‖A − Â‖₂")
    ax.grid(True)
    ax.legend(fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# %% [markdown]
# ## 4. A single-pass sketch: compressing what you can only see once
#
# The range finder reads $A$ twice ($A\Omega$, then $Q^{\mathsf T}A$). For streaming data, or a matrix
# assembled from updates you cannot store, even that is too much. **Tropp, Yurtsever, Udell and Cevher
# (2017)** keep two sketches built in a single pass — a range sketch $Y = A\Omega$ and a co-range
# sketch $W = \Psi A$ with $\Psi$ a second random matrix — and reconstruct
#
# $$A \;\approx\; Q\,(\Psi Q)^{\dagger}\,W, \qquad Q = \operatorname{orth}(Y).$$
#
# Because both sketches are linear in $A$, they can be updated as entries of $A$ arrive: $A \to
# A + H$ just adds $H\Omega$ and $\Psi H$. Nothing else is ever stored.

# %%
A = A_fast
for k in (15, 25, 35):
    l = 2 * k + 1  # co-range sketch somewhat larger, as their analysis recommends
    Om = rng.normal(size=(n, k))
    Psi = rng.normal(size=(l, n))
    Y_s, W_s = A @ Om, Psi @ A
    Qs, _ = np.linalg.qr(Y_s)
    A_hat = Qs @ np.linalg.lstsq(Psi @ Qs, W_s, rcond=None)[0]
    print(
        f"k = {k:2d}: ‖A − Â‖₂ = {np.linalg.norm(A - A_hat, 2):.2e}"
        f"   optimal σ_(k+1) = {s_fast[k]:.2e}   ratio {np.linalg.norm(A - A_hat, 2) / s_fast[k]:5.1f}"
        f"   stored {Y_s.size + W_s.size:,} of {A.size:,} numbers"
    )

# linearity: the sketch of a sum is the sum of the sketches, so it can absorb a stream of updates
updates = [rng.normal(size=(n, n)) * 1e-3 for _ in range(5)]
Y_stream, W_stream = A @ Om, Psi @ A
for H in updates:
    Y_stream, W_stream = Y_stream + H @ Om, W_stream + Psi @ H
A_total = A + sum(updates)
print(
    "\nsketch updated incrementally == sketch of the final matrix :",
    np.allclose(Y_stream, A_total @ Om) and np.allclose(W_stream, Psi @ A_total),
)

# %% [markdown]
# The honest reading: a single pass costs accuracy. At these sketch sizes the error is 5–10× the
# optimum rather than equal to it — the price of never revisiting $A$ — and the gap is why Tropp et al.
# recommend sketches somewhat larger than the target rank. In exchange, memory is a
# fixed fraction of the matrix, and the sketch absorbs updates for free.
#
# ## 5. Cross approximation: building the low-rank form from a few function evaluations
#
# Sections 3 and 4 need *products* with $A$. For a function we can only *evaluate*, there is a more
# direct route: if $f$ has rank $r$, it is determined by **$r$ of its rows and $r$ of its columns**.
# Choosing rows $I$ and columns $J$,
#
# $$A \;\approx\; A_{:,J}\,\big(A_{I,J}\big)^{-1}A_{I,:}\qquad\text{(skeleton / CUR approximation)},$$
#
# which is exact when $A$ has rank $r$ and $A_{I,J}$ is invertible. The quality depends on choosing
# $A_{I,J}$ of large volume (Goreinov, Tyrtyshnikov and Zamarashkin 1997). **Adaptive cross
# approximation** (Bebendorf 2000) chooses the pivots greedily: take a row of the current residual,
# pivot on its largest entry, subtract the rank-one cross through that pivot, and repeat. Each step
# evaluates $f$ along one row and one column — $m+n$ samples — instead of all $mn$.


# %%
def aca(f, xs, ys, tol=1e-10, rmax=200):
    """Adaptive cross approximation with partial pivoting. Returns U, V, pivots, evaluations used."""
    m, nn = len(xs), len(ys)
    Us, Vs, pivots, used = [], [], [], set()
    i, evals, est2 = 0, 0, 0.0
    for _ in range(rmax):
        row = f(xs[i], ys) - sum(u[i] * v for u, v in zip(Us, Vs))  # residual row i
        evals += nn
        j = int(np.argmax(np.abs(row)))
        if abs(row[j]) < 1e-300:
            break
        v = row / row[j]
        col = f(xs, ys[j]) - sum(
            u * vv[j] for u, vv in zip(Us, Vs)
        )  # residual column j
        evals += m
        Us.append(col)
        Vs.append(v)
        pivots.append((i, j))
        used.add(i)
        step = np.linalg.norm(col) * np.linalg.norm(v)
        est2 += step**2
        if step <= tol * np.sqrt(est2):  # the new cross is negligible: stop
            break
        cand = np.abs(col)
        cand[list(used)] = -1.0
        i = int(np.argmax(cand))  # next row: where the new column is largest
    return np.array(Us).T, np.array(Vs), pivots, evals


cases = [
    ("1/(1+x+y)", lambda a, b: 1.0 / (1.0 + a + b), examples["1/(1+x+y)"]),
    (
        "exp(-(x-y)²/0.01)",
        lambda a, b: np.exp(-((a - b) ** 2) / 0.01),
        examples["exp(-(x-y)²/0.01)"],
    ),
]
aca_results = {}
for name, f, Afull in cases:
    Uc, Vc, piv, ev = aca(f, x, x)
    err = np.max(np.abs(Uc @ Vc - Afull)) / np.max(np.abs(Afull))
    aca_results[name] = piv
    print(
        f"{name:>18}: rank {Uc.shape[1]:3d}   {ev:7,d} evaluations of {n * n:,}"
        f" ({ev / (n * n):5.1%})   max relative error {err:.1e}"
    )

# The skeleton formula, checked — and a numerical lesson. A[I,J] is mathematically invertible but
# terribly conditioned (its entries are samples of a smooth function), so HOW it is inverted matters.
piv = aca_results["1/(1+x+y)"]
I_idx, J_idx = [p_[0] for p_ in piv], [p_[1] for p_ in piv]
Af = examples["1/(1+x+y)"]
AIJ = Af[np.ix_(I_idx, J_idx)]
skel_err = lambda B: np.max(np.abs(B - Af)) / Af.max()
print(
    f"\nskeleton A[:,J] A[I,J]⁻¹ A[I,:] from the {len(I_idx)} pivot rows/columns; cond(A[I,J]) = {np.linalg.cond(AIJ):.1e}"
)
print(
    f"   via an explicit pseudo-inverse : max rel error {skel_err(Af[:, J_idx] @ np.linalg.pinv(AIJ, rcond=1e-15) @ Af[I_idx, :]):.1e}"
)
print(
    f"   via an LU solve                : max rel error {skel_err(Af[:, J_idx] @ np.linalg.solve(AIJ, Af[I_idx, :])):.1e}"
)
print(
    "   ACA's residual updates ARE Gaussian elimination on A[I,J], one pivot at a time — which is"
)
print(
    "   why ACA reached 1e-12 while the textbook formula, inverted naively, loses eight digits."
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
fig.suptitle(
    "Adaptive cross approximation: the pivots it chose, and the only rows and columns it ever evaluated",
    color="white",
    fontsize=11,
)
for ax, (name, _, Afull) in zip(axes, cases):
    piv = aca_results[name]
    ax.imshow(Afull.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", alpha=0.85)
    for i_, j_ in piv:
        ax.axvline(x[i_], color=BLUE, lw=0.6, alpha=0.5)
        ax.axhline(x[j_], color=GREEN, lw=0.6, alpha=0.5)
    ax.plot([x[p[0]] for p in piv], [x[p[1]] for p in piv], "o", color=YELLOW, ms=5)
    ax.set_title(f"{name}: {len(piv)} crosses", color="#ccc", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# For $1/(1+x+y)$ the whole $400\times400$ function is recovered to twelve digits from 4% of its
# values, and the cost grows like $r(m+n)$, not $mn$. The pivots for the Gaussian ridge sit along the
# diagonal — exactly where the function's information is — and the method found them without being
# told. This is the version of "sketching" that matters when the function is expensive to evaluate:
# a simulator, a likelihood, a kernel over a large dataset. It is also the engine of Chebfun2
# (Townsend & Trefethen 2013), which represents bivariate functions this way automatically.
#
# A caution worth stating: partial-pivoting ACA has no worst-case guarantee — a function that is zero
# on every row it happens to sample can fool it. In practice it is remarkably reliable for smooth
# functions; the guaranteed alternatives (rook pivoting, randomized row selection with leverage
# scores, Mahoney & Drineas 2009) cost more.
#
# ## 6. Three variables: CP and Tucker, and how tensor rank misbehaves
#
# A function of three variables on a grid is an order-3 tensor $T_{ijk} = f(x_i,y_j,z_k)$, and the
# matrix story splits in two, because there is no single "SVD of a tensor."
#
# * **CP** (canonical polyadic; Hitchcock 1927, rediscovered as CANDECOMP by Carroll & Chang and as
#   PARAFAC by Harshman, both 1970): $T \approx \sum_{k=1}^r \mathbf u_k\otimes\mathbf v_k\otimes\mathbf
#   w_k$, the direct generalization of a sum of separable terms. Storage $3nr$.
# * **Tucker** (Tucker 1966; computed by the higher-order SVD of De Lathauwer, De Moor & Vandewalle
#   2000): $T \approx G\times_1 U\times_2 V\times_3 W$, a small core $G$ of size $r_1\times r_2\times
#   r_3$ with an orthonormal basis for each variable. Its ranks are just the matrix ranks of the three
#   *unfoldings* of $T$, so it inherits all the good behaviour of the SVD.
#
# The function $\sin(x+y+z)$ shows how differently these behave. By the addition formula it lives in
# $\{\sin,\cos\}^{\otimes3}$, so its Tucker rank is $(2,2,2)$. Its CP rank is subtler.

# %%
nt = 30
t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
X3, Y3, Z3 = np.meshgrid(t, t, t, indexing="ij")
T_sin = np.sin(X3 + Y3 + Z3)

# Tucker / multilinear rank: the ranks of the three unfoldings
print(
    "ranks of the unfoldings of sin(x+y+z):",
    [
        int(np.linalg.matrix_rank(tl.unfold(tl.tensor(T_sin), m), tol=1e-10))
        for m in range(3)
    ],
    " → Tucker rank (2,2,2)",
)
core, factors = tucker(tl.tensor(T_sin), rank=[2, 2, 2])
print(
    "Tucker (2,2,2) relative error:",
    f"{np.linalg.norm(tl.tucker_to_tensor((core, factors)) - T_sin) / np.linalg.norm(T_sin):.1e}",
)

# Over ℂ the CP rank is 2: sin(s) = (e^{is} − e^{−is}) / 2i, and each exponential is separable.
e = np.exp(1j * t)
outer3 = lambda a, b, c: np.einsum("i,j,k->ijk", a, b, c)
T_complex2 = ((outer3(e, e, e) - outer3(e.conj(), e.conj(), e.conj())) / 2j).real
print(
    "complex CP rank 2 (two exponentials) reproduces it:",
    np.allclose(T_complex2, T_sin),
)

# Over ℝ: fit CP with ALS at ranks 1, 2, 3, keeping the best of several random starts.
print("\nreal CP fits by alternating least squares (best of 8 starts):")
for r in (1, 2, 3):
    best = min(
        np.linalg.norm(
            tl.cp_to_tensor(
                parafac(
                    tl.tensor(T_sin),
                    rank=r,
                    init="random",
                    random_state=sd,
                    n_iter_max=3000,
                    tol=1e-14,
                )
            )
            - T_sin
        )
        / np.linalg.norm(T_sin)
        for sd in range(8)
    )
    print(f"   rank {r}: relative error {best:.2e}")

# %% [markdown]
# Over the complex numbers two terms suffice; over the reals two terms leave **half** the norm
# unexplained and three are exact. The real CP rank is 3. This is not an optimization failure, and it
# can be certified exactly. In the basis $\{\sin,\cos\}$ the function *is* a $2\times2\times2$
# tensor, and a real $2\times2\times2$ tensor has rank 2 or 3 according to the sign of **Cayley's
# hyperdeterminant** $\Delta$ — rank 2 when $\Delta>0$, rank 3 when $\Delta<0$ (de Silva & Lim 2008).
# We compute $\Delta$, and to avoid trusting the sign convention we check it against ALS on random
# tensors.


# %%
def hyperdet(a):
    """Cayley's hyperdeterminant of a 2×2×2 array."""
    return (
        a[0, 0, 0] ** 2 * a[1, 1, 1] ** 2
        + a[0, 0, 1] ** 2 * a[1, 1, 0] ** 2
        + a[0, 1, 0] ** 2 * a[1, 0, 1] ** 2
        + a[1, 0, 0] ** 2 * a[0, 1, 1] ** 2
        - 2
        * (
            a[0, 0, 0] * a[0, 0, 1] * a[1, 1, 0] * a[1, 1, 1]
            + a[0, 0, 0] * a[0, 1, 0] * a[1, 0, 1] * a[1, 1, 1]
            + a[0, 0, 0] * a[1, 0, 0] * a[0, 1, 1] * a[1, 1, 1]
            + a[0, 0, 1] * a[0, 1, 0] * a[1, 0, 1] * a[1, 1, 0]
            + a[0, 0, 1] * a[1, 0, 0] * a[0, 1, 1] * a[1, 1, 0]
            + a[0, 1, 0] * a[1, 0, 0] * a[0, 1, 1] * a[1, 0, 1]
        )
        + 4
        * (
            a[0, 0, 0] * a[0, 1, 1] * a[1, 0, 1] * a[1, 1, 0]
            + a[0, 0, 1] * a[0, 1, 0] * a[1, 0, 0] * a[1, 1, 1]
        )
    )


def best_rank2_error(a, starts=6):
    return min(
        np.linalg.norm(
            tl.cp_to_tensor(
                parafac(
                    tl.tensor(a),
                    rank=2,
                    init="random",
                    random_state=sd,
                    n_iter_max=2000,
                    tol=1e-15,
                )
            )
            - a
        )
        / np.linalg.norm(a)
        for sd in range(starts)
    )


# sin(x+y+z) = sin x cos y cos z + cos x sin y cos z + cos x cos y sin z − sin x sin y sin z
C = np.zeros((2, 2, 2))  # index 0 = sin, 1 = cos
C[0, 1, 1] = C[1, 0, 1] = C[1, 1, 0] = 1.0
C[0, 0, 0] = -1.0
Sb = np.stack([np.sin(t), np.cos(t)], axis=1)
print(
    "sin(x+y+z) equals this 2×2×2 tensor in the basis {sin, cos}:",
    np.allclose(np.einsum("abc,ia,jb,kc->ijk", C, Sb, Sb, Sb), T_sin),
)
print(f"its hyperdeterminant Δ = {hyperdet(C):+.1f}  →  real rank 3\n")

# Checking the sign convention on random tensors, rather than trusting it:
agree = 0
for sd in range(12):
    a = np.random.default_rng(100 + sd).normal(size=(2, 2, 2))
    predicted_rank2 = hyperdet(a) > 0
    fits_rank2 = best_rank2_error(a) < 1e-6
    agree += predicted_rank2 == fits_rank2
print(
    f"sign of Δ predicts whether a rank-2 fit is exact for {agree} of 12 random tensors"
)

# %% [markdown]
# So the rank of the *same* function depends on the number field and on the format: Tucker rank
# $(2,2,2)$, complex CP rank 2, real CP rank 3. A matrix's rank is the same over $\mathbb R$ and
# $\mathbb C$; a tensor's need not be.
#
# ### The best rank-$r$ CP approximation may not exist
#
# It gets stranger. For matrices, Eckart–Young guarantees a best rank-$r$ approximation. For tensors
# there may be **none**: the set of tensors of rank $\le r$ is not closed, so a sequence of rank-2
# tensors can converge to a rank-3 one (de Silva & Lim 2008). The standard example is
#
# $$W = e_0\otimes e_0\otimes e_1 + e_0\otimes e_1\otimes e_0 + e_1\otimes e_0\otimes e_0
#     = \lim_{\varepsilon\to0}\tfrac{1}{\varepsilon}\Big[(e_0+\varepsilon e_1)^{\otimes3} - e_0^{\otimes3}\Big],$$
#
# rank 3, but approximable arbitrarily well by rank 2 (its *border rank* is 2). The price is visible:
# the two terms grow like $1/\varepsilon$ and cancel. Hand ALS this tensor and it does the same.

# %%
e0, e1 = np.eye(2)
W = outer3(e0, e0, e1) + outer3(e0, e1, e0) + outer3(e1, e0, e0)
print(
    f"hyperdeterminant of W: {hyperdet(W):+.1f}  (zero: on the boundary between rank 2 and rank 3)\n"
)
print("explicit rank-2 sequence:")
for eps in (1e-1, 1e-2, 1e-3):
    a_ = e0 + eps * e1
    approx = (outer3(a_, a_, a_) - outer3(e0, e0, e0)) / eps
    print(
        f"   ε = {eps:.0e}:  ‖approx − W‖ = {np.linalg.norm(approx - W):.1e}   size of each term ≈ {1 / eps:.0e}"
    )
print("\nALS asked for the best rank-2 fit of W:")
for iters in (50, 500, 5000):
    cp = parafac(
        tl.tensor(W), rank=2, init="random", random_state=1, n_iter_max=iters, tol=0
    )
    err = np.linalg.norm(tl.cp_to_tensor(cp) - W) / np.linalg.norm(W)
    size = max(
        abs(cp.weights[r_]) * np.prod([np.linalg.norm(fm[:, r_]) for fm in cp.factors])
        for r_ in range(2)
    )
    print(
        f"   {iters:5d} iterations: relative error {err:.2e}   largest term norm {size:.2f}"
    )
print(
    "\nThe error keeps shrinking and the terms keep growing: there is no minimizer to converge to."
)
print(
    "This 'degeneracy' is the classic failure of CP fitting in practice (diverging, cancelling"
)
print(
    "components). Tucker, being built from SVDs of unfoldings, never has this problem."
)

# %% [markdown]
# **Practical upshot.** CP is the most compact and the most interpretable format — each term is a
# product of one-variable profiles, which is why it is the format of choice for finding components in
# data — but its rank can depend on the field, its best approximation can fail to exist, and fitting
# it is a nonconvex problem with degenerate solutions. Tucker is stable and computable by SVDs but
# its core grows like $r^d$. Section 7 resolves that tension for many variables.
#
# ## 7. Many variables: tensor trains and the curse of dimensionality
#
# With $d$ variables and $n$ grid points each, the raw tensor has $n^d$ entries — $10^{20}$ for
# $d=n=20$ — and even a Tucker core needs $r^d$. The **tensor train** (Oseledets 2011; the same
# object as the *matrix product states* of quantum many-body physics) writes
#
# $$T_{i_1\cdots i_d} = G_1[i_1]\,G_2[i_2]\cdots G_d[i_d],$$
#
# a product of small matrices, each $G_k[i_k]$ of size $r_{k-1}\times r_k$. Storage is $\sum_k
# n\,r_{k-1}r_k$ — **linear** in $d$. The TT ranks are the ranks of the sequential unfoldings, and TT-SVD
# computes them with ordinary SVDs, so it is as well-posed as Tucker and as economical as CP.


# %%
def tt_svd(T, eps=1e-12):
    """Tensor-train decomposition by sequential truncated SVDs (Oseledets 2011, Alg. 1)."""
    d, shape = T.ndim, T.shape
    delta = eps * np.linalg.norm(T) / np.sqrt(d - 1)
    cores, r, Cm = [], 1, T.reshape(1, -1)
    for k in range(d - 1):
        Cm = Cm.reshape(r * shape[k], -1)
        Uk, sk, Vtk = np.linalg.svd(Cm, full_matrices=False)
        tail = np.sqrt(np.cumsum(sk[::-1] ** 2))[::-1]
        rk = max(1, int(np.sum(tail > delta)))
        cores.append(Uk[:, :rk].reshape(r, shape[k], rk))
        Cm, r = sk[:rk, None] * Vtk[:rk], rk
    cores.append(Cm.reshape(r, shape[-1], 1))
    return cores


def tt_full(cores):
    out = cores[0]
    for G in cores[1:]:
        out = np.tensordot(out, G, axes=([-1], [0]))
    return out.reshape(out.shape[1:-1])


d, nd = 6, 12
g = np.linspace(0, 1, nd)
grids = np.meshgrid(*([g] * d), indexing="ij")
S_sum = sum(grids)
print(f"d = {d} variables, {nd} points each: the full tensor has {nd**d:,} entries\n")
print(f"{'f(x₁,…,x₆)':>18}{'TT ranks':>22}{'TT storage':>12}{'rel. error':>12}")
for name, Tm in [
    ("exp(−Σ xᵢ²)", np.exp(-sum(G_**2 for G_ in grids))),
    ("sin(Σ xᵢ)", np.sin(S_sum)),
    ("1/(1 + Σ xᵢ)", 1.0 / (1.0 + S_sum)),
]:
    cores = tt_svd(Tm, 1e-10)
    err = np.linalg.norm(tt_full(cores) - Tm) / np.linalg.norm(Tm)
    print(
        f"{name:>18}{str([c.shape[2] for c in cores[:-1]]):>22}{sum(c.size for c in cores):>12,}{err:>12.1e}"
    )

# %% [markdown]
# The middle row is the punchline of this section. $\sin(x_1+\cdots+x_d)$ has **TT rank 2 for every
# $d$** — each unfolding is a function of a partial sum $a$ and its complement $b$, and $\sin(a+b) =
# \sin a\cos b + \cos a\sin b$ — while its real CP rank grows with $d$ (Section 6 found 3 already at
# $d=3$). So the format matters as much as the function: the right format turns an object of
# $n^d$ numbers into one of $4nd$.
#
# The plot below shows what that means as $d$ grows, with the TT rank held at the value just measured.

# %%
dims = np.arange(2, 41)
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.semilogy(dims, float(nd) ** dims, "-", color=RED, lw=2.5, label="full grid  n^d")
ax.semilogy(
    dims,
    [nd * 8.0**2 * dd for dd in dims],
    "-",
    color=ORANGE,
    lw=2,
    label="TT, rank 8:  ≈ n·r²·d",
)
ax.semilogy(
    dims,
    [nd * 2.0**2 * dd for dd in dims],
    "-",
    color=GREEN,
    lw=2,
    label="TT, rank 2 (sin Σxᵢ)",
)
ax.axhline(1e12, color=GREY, ls=":", lw=1)
ax.text(2.5, 2e12, "≈ 8 TB of float64", color=GREY, fontsize=8)
ax.set_xlabel("number of variables d   (n = 12 points each)")
ax.set_ylabel("numbers stored")
ax.set_title(
    "The curse of dimensionality, and the way around it", color="#ccc", fontsize=10
)
ax.grid(True, which="both")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Two honest caveats. TT-SVD as written still needs the full tensor in memory, so it cannot by itself
# beat the curse; the methods that can — **TT-cross** (Oseledets & Tyrtyshnikov 2010), the tensor
# analogue of Section 5 — build the cores from a few sampled fibres, exactly as ACA built a matrix
# from a few rows and columns. And TT ranks depend on the *ordering* of the variables: strongly
# coupled variables should sit next to each other in the train.
#
# ## 8. Where this shows up
#
# * **EEG and MEG.** A recording is naturally a tensor — channels × time (or frequency) × trials or
#   subjects — and a CP/PARAFAC decomposition returns components that are each a spatial map times a
#   time course times a trial profile. Its uniqueness under mild conditions (Kruskal 1977), which
#   matrix factorizations lack, is why it is used to extract interpretable rhythms: Miwakeichi et al.
#   (2004) decomposed EEG into space–time–frequency atoms this way, and Mørup et al. (2006) applied it
#   to wavelet-transformed event-related EEG. The degeneracy of Section 6 is a practical hazard there;
#   non-negativity constraints are the usual remedy.
# * **Compressing neural networks.** A dense weight matrix is replaced by a product of thin factors —
#   which is precisely how LoRA adapts large models (Hu et al. 2022) — and convolution kernels, which
#   are order-4 tensors, are replaced by their CP decompositions (Lebedev et al. 2015).
# * **Latent-space geometry.** The singular-value spectrum of a layer's Jacobian or of a matrix of
#   embeddings is exactly the decay curve of Section 1. Its effective rank is a measure of how many
#   directions a representation really uses, and the randomized SVD of Section 3 is how that spectrum
#   is computed when the matrix is too large to form. Udell & Townsend (2019) give a nice argument for
#   why large data matrices are so often approximately low rank in the first place.
#
# **The thread through all of it:** structure means *few separable terms in the right coordinates and
# the right format*, and randomness or sampling finds those terms without ever paying for the full
# object.
#
# ## References
#
# **Low-rank approximation and why it works**
#
# - C. Eckart and G. Young, "The approximation of one matrix by another of lower rank",
#   *Psychometrika* **1** (1936), 211–218.
# - B. Beckermann and A. Townsend, "On the singular values of matrices with displacement structure",
#   *SIAM J. Matrix Anal. Appl.* **38** (2017), 1227–1248.
# - M. Udell and A. Townsend, "Why are big data matrices approximately low rank?", *SIAM J. Math. Data
#   Sci.* **1** (2019), 144–160.
#
# **Sketching**
#
# - W. B. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space",
#   *Contemp. Math.* **26** (1984), 189–206.
# - N. Halko, P.-G. Martinsson and J. A. Tropp, "Finding structure with randomness: probabilistic
#   algorithms for constructing approximate matrix decompositions", *SIAM Review* **53** (2011),
#   217–288. The place to start.
# - J. A. Tropp, A. Yurtsever, M. Udell and V. Cevher, "Practical sketching algorithms for low-rank
#   matrix approximation", *SIAM J. Matrix Anal. Appl.* **38** (2017), 1454–1485.
# - D. P. Woodruff, "Sketching as a tool for numerical linear algebra", *Found. Trends Theor. Comput.
#   Sci.* **10** (2014), 1–157.
# - P.-G. Martinsson and J. A. Tropp, "Randomized numerical linear algebra: foundations and
#   algorithms", *Acta Numerica* **29** (2020), 403–572. The current comprehensive survey.
#
# **Cross / skeleton approximation**
#
# - S. A. Goreinov, E. E. Tyrtyshnikov and N. L. Zamarashkin, "A theory of pseudoskeleton
#   approximations", *Linear Algebra Appl.* **261** (1997), 1–21.
# - M. Bebendorf, "Approximation of boundary element matrices", *Numer. Math.* **86** (2000), 565–589.
# - M. W. Mahoney and P. Drineas, "CUR matrix decompositions for improved data analysis", *PNAS*
#   **106** (2009), 697–702.
# - A. Townsend and L. N. Trefethen, "An extension of Chebfun to two dimensions", *SIAM J. Sci.
#   Comput.* **35** (2013), C495–C518.
#
# **Tensors**
#
# - F. L. Hitchcock, "The expression of a tensor or a polyadic as a sum of products", *J. Math. Phys.*
#   **6** (1927), 164–189.
# - L. R. Tucker, "Some mathematical notes on three-mode factor analysis", *Psychometrika* **31**
#   (1966), 279–311.
# - J. D. Carroll and J.-J. Chang (1970), *Psychometrika* **35**, 283–319; R. A. Harshman (1970),
#   *UCLA Working Papers in Phonetics* **16**, 1–84 — CANDECOMP and PARAFAC, independently.
# - J. B. Kruskal, "Three-way arrays: rank and uniqueness of trilinear decompositions", *Linear
#   Algebra Appl.* **18** (1977), 95–138.
# - L. De Lathauwer, B. De Moor and J. Vandewalle, "A multilinear singular value decomposition",
#   *SIAM J. Matrix Anal. Appl.* **21** (2000), 1253–1278.
# - V. de Silva and L.-H. Lim, "Tensor rank and the ill-posedness of the best low-rank approximation
#   problem", *SIAM J. Matrix Anal. Appl.* **30** (2008), 1084–1127.
# - T. G. Kolda and B. W. Bader, "Tensor decompositions and applications", *SIAM Review* **51**
#   (2009), 455–500. The standard survey.
# - I. V. Oseledets, "Tensor-train decomposition", *SIAM J. Sci. Comput.* **33** (2011), 2295–2317;
#   I. Oseledets and E. Tyrtyshnikov, "TT-cross approximation for multidimensional arrays", *Linear
#   Algebra Appl.* **432** (2010), 70–88.
#
# **Applications**
#
# - F. Miwakeichi et al., "Decomposing EEG data into space–time–frequency components using Parallel
#   Factor Analysis", *NeuroImage* **22** (2004), 1035–1045.
# - M. Mørup et al., "Parallel factor analysis as an exploratory tool for wavelet transformed
#   event-related EEG", *NeuroImage* **29** (2006), 938–947.
# - V. Lebedev et al., "Speeding-up convolutional neural networks using fine-tuned CP-decomposition",
#   *ICLR* 2015.
# - E. J. Hu et al., "LoRA: Low-rank adaptation of large language models", *ICLR* 2022.
