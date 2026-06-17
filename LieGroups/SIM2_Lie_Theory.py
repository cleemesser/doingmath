# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Complex Numbers and SIM(2)
#
# This notebook braids two stories together:
#
# 1. **Complex numbers as $2\times2$ matrices.** The map $a + bi \mapsto \left(\begin{smallmatrix} a & -b \\ b & a \end{smallmatrix}\right)$
#    turns complex multiplication into matrix multiplication. This extends the SO(2) $\cong$ U(1)
#    story from the *unit circle* to *all* of $\mathbb{C}$: the nonzero complex numbers
#    $\mathbb{C}^\times$ are exactly the **scale-rotations** of the plane, and the complex
#    exponential $e^{\lambda + i\omega} = e^{\lambda}(\cos\omega + i\sin\omega)$ *is* the matrix
#    exponential of $\left(\begin{smallmatrix}\lambda & -\omega \\ \omega & \lambda\end{smallmatrix}\right)$.
#
# 2. **SIM(2): the similarity group of the plane.** Orientation-preserving similarities are the
#    complex affine maps $z \mapsto a z + b$ with $a \in \mathbb{C}^\times$ (rotation + uniform
#    scale) and $b \in \mathbb{C}$ (translation). This is the 4-dimensional Lie group
#    $\mathbb{C}^\times \ltimes \mathbb{C}$ — SE(2) with an extra scaling degree of freedom — and
#    *its entire Lie theory collapses into complex arithmetic*.
#
# We illustrate the group, the Lie algebra $\mathfrak{sim}(2)$, the bracket (**ad**), the adjoint
# (**Ad**), the relation $\text{Ad}_{e^X} = e^{\text{ad}_X}$, and the signature SIM(2) geometry:
# the **logarithmic spiral**.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

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


# %% [markdown]
# ## 1. Complex Numbers as $2\times2$ Matrices
#
# Represent $z = a + bi$ by the real matrix
#
# $$\Phi(z) = \begin{pmatrix} a & -b \\ b & a \end{pmatrix} = a\,I + b\,G, \qquad G = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}.$$
#
# Then $\Phi$ is a **ring isomorphism** onto these matrices: complex addition/multiplication become
# matrix addition/multiplication, $|z|^2 = \det\Phi(z)$, and — crucially — the complex exponential
# equals the matrix exponential. Writing $z = \lambda + i\omega$ in "polar-log" form,
# $\Phi(z) = \lambda I + \omega G$ exponentiates to $e^{\lambda} R(\omega)$: **scale by $e^\lambda$,
# rotate by $\omega$**. (Restricting to $\lambda = 0$, $|z| = 1$, recovers exactly SO(2) $\cong$ U(1).)

# %%
G = np.array([[0.0, -1.0], [1.0, 0.0]])


def C2M(z):
    """Represent a complex number as a 2x2 real matrix Φ(z) = Re(z)·I + Im(z)·G."""
    return z.real * np.eye(2) + z.imag * G


z1, z2 = 1.5 + 0.8j, -0.4 + 1.2j
print("Φ is a ring homomorphism:")
print("  Φ(z1·z2) == Φ(z1)·Φ(z2):", np.allclose(C2M(z1 * z2), C2M(z1) @ C2M(z2)))
print("  Φ(z1+z2) == Φ(z1)+Φ(z2):", np.allclose(C2M(z1 + z2), C2M(z1) + C2M(z2)))
print("  |z|² == det Φ(z):", np.isclose(abs(z1) ** 2, np.linalg.det(C2M(z1))))

# complex exponential == matrix exponential
z = 0.3 + 1.1j  # λ = 0.3 (scale-log), ω = 1.1 (rotation)
print("\nComplex exp == matrix exp of Φ(z):")
print("  e^z =", np.round(np.exp(z), 4))
print("  Φ(e^z) =\n", np.round(C2M(np.exp(z)), 4))
print("  expm(Φ(z)) =\n", np.round(expm(C2M(z)), 4))
print("  match:", np.allclose(C2M(np.exp(z)), expm(C2M(z))))
print(
    "  → scale by e^λ =",
    round(np.exp(z.real), 4),
    " and rotate by ω =",
    round(z.imag, 4),
)

# --- Visualize: multiplying by a ∈ ℂ* scales AND rotates ---
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.set_title(
    "Multiplication by a = 1.3·e^{i·40°} ∈ ℂˣ: each step scales ×1.3 and rotates 40°",
    color="#ccc",
    fontsize=10,
)
ax.set_aspect("equal")
ax.grid(True)
ax.axhline(0, color="#555", lw=0.8)
ax.axvline(0, color="#555", lw=0.8)
a = 1.3 * np.exp(1j * np.deg2rad(40))
zk = 1.0 + 0j
pts = [zk]
for _ in range(7):
    zk = a * zk
    pts.append(zk)
pts = np.array(pts)
ax.plot(pts.real, pts.imag, "-o", color=BLUE, lw=1.5, ms=5)
for k, p in enumerate(pts):
    ax.annotate(
        f"aᵏz, k={k}" if k in (0, 7) else "", (p.real, p.imag), color="#aaa", fontsize=8
    )
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. SIM(2): The Similarity Group
#
# An orientation-preserving similarity of the plane is a complex affine map
#
# $$g = (a, b): \quad z \longmapsto a\,z + b, \qquad a \in \mathbb{C}^\times,\; b \in \mathbb{C}.$$
#
# Here $a = s\,e^{i\theta}$ rotates by $\theta$ and scales by $s = |a|$, and $b$ translates.
# Composition multiplies the complex $2\times2$ matrices $\left(\begin{smallmatrix} a & b \\ 0 & 1 \end{smallmatrix}\right)$
# acting on $\left(\begin{smallmatrix} z \\ 1 \end{smallmatrix}\right)$, giving the semidirect-product law
# $(a_1, b_1)(a_2, b_2) = (a_1 a_2,\; a_1 b_2 + b_1)$. Equivalently, as **real $3\times3$**
# homogeneous matrices (SE(2) with the rotation block $R$ replaced by the scale-rotation $\Phi(a)$):
#
# $$M(a, b) = \begin{pmatrix} \Phi(a) & (\mathrm{Re}\,b,\ \mathrm{Im}\,b)^\top \\ 0 & 1 \end{pmatrix}.$$


# %%
def sim2_real(a, b):
    """SIM(2) element as a real 3x3 homogeneous matrix; a, b are complex."""
    M = np.eye(3)
    M[:2, :2] = C2M(a)
    M[:2, 2] = [b.real, b.imag]
    return M


def sim2_complex(a, b):
    """SIM(2) element as a complex 2x2 matrix [[a, b], [0, 1]] acting on (z, 1)."""
    return np.array([[a, b], [0, 1]], dtype=complex)


def sim2_inverse(a, b):
    """Inverse of (a, b): z ↦ az+b has inverse z ↦ (1/a)z − b/a."""
    return 1 / a, -b / a


def apply_real(M, z):
    """Apply a real 3x3 SIM(2) matrix to a complex point z."""
    out = M @ np.array([z.real, z.imag, 1.0])
    return out[0] + 1j * out[1]


# the real and complex representations agree, and composition works
g1, g2 = (1.4 * np.exp(1j * 0.6), 0.5 + 0.3j), (0.8 * np.exp(1j * 1.1), -0.7 + 0.4j)
z0 = 0.9 - 0.2j
print(
    "Real 3×3 and complex z↦az+b agree on a point:",
    np.isclose(apply_real(sim2_real(*g1), z0), g1[0] * z0 + g1[1]),
)
# composition law (a1a2, a1 b2 + b1)
comp = sim2_real(*g1) @ sim2_real(*g2)
a12, b12 = g1[0] * g2[0], g1[0] * g2[1] + g1[1]
print(
    "Composition (a1a2, a1b2+b1) matches matrix product:",
    np.allclose(comp, sim2_real(a12, b12)),
)
print(
    "Non-commutative (g1g2 ≠ g2g1):",
    not np.allclose(sim2_real(*g1) @ sim2_real(*g2), sim2_real(*g2) @ sim2_real(*g1)),
)
print(
    "Inverse check g·g⁻¹ = I:",
    np.allclose(sim2_real(*g1) @ sim2_real(*sim2_inverse(*g1)), np.eye(3)),
)

# --- Visualize a similarity transform of a shape ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "SIM(2): rigid motion + uniform scaling of the plane", color="white", fontsize=13
)
# an 'F' so orientation and chirality are visible
F = np.array([0, 0.6, 0.6, 0.3, 0.3, 0.5, 0.5, 0.3, 0.3, 0]) + 1j * np.array(
    [0, 0, 0.12, 0.12, 0.45, 0.45, 0.57, 0.57, 1.0, 1.0]
)
for ax, (a, b, title) in zip(
    axes,
    [
        (1.0 * np.exp(1j * 0.0), 0 + 0j, "original  (a=1, b=0)"),
        (
            1.6 * np.exp(1j * np.deg2rad(50)),
            1.0 + 0.4j,
            "a = 1.6·e^{i·50°},  b = 1.0+0.4i",
        ),
    ],
):
    w = a * F + b
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.grid(True)
    ax.fill(F.real, F.imag, alpha=0.2, color="white")
    ax.plot(F.real, F.imag, "--", color="white", alpha=0.4)
    ax.fill(w.real, w.imag, alpha=0.4, color=ORANGE)
    ax.plot(w.real, w.imag, "-", color=ORANGE, lw=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The Lie Algebra $\mathfrak{sim}(2)$ and the Spiral
#
# $\mathfrak{sim}(2)$ is 4-dimensional, with generators **scale** $S$, **rotation** $J$, and two
# **translations** $P_1, P_2$. A general element pairs a complex "scale-rotation rate"
# $\alpha = \lambda + i\omega$ with a complex translation rate $\beta = u_x + i u_y$:
#
# $$X(\alpha, \beta) = \begin{pmatrix} \lambda & -\omega & u_x \\ \omega & \lambda & u_y \\ 0 & 0 & 0 \end{pmatrix}
#   \;\longleftrightarrow\; \begin{pmatrix} \alpha & \beta \\ 0 & 0 \end{pmatrix} \ (\text{complex}).$$
#
# The exponential has a **closed form in pure complex arithmetic** — the analogue of SE(2)'s
# left-Jacobian, but here it is just a *complex number* $V(\alpha) = (e^{\alpha} - 1)/\alpha$:
#
# $$e^{X(\alpha,\beta)} = \big(a, b\big), \qquad a = e^{\alpha}, \quad b = V(\alpha)\,\beta.$$
#
# Because $a = e^{\alpha}$ both rotates and scales, a one-parameter subgroup $e^{tX}$ sweeps a
# **logarithmic spiral** — SIM(2)'s geometric signature.
#
# > **Caveat — closed forms run out.** SIM(2), like SO(3)/SE(3), is lucky: its exp closes in
# > elementary functions (here even in scalar complex arithmetic). A generic Lie algebra has no
# > such formula and one sums the matrix-exponential series via `scipy.linalg.expm` — which we
# > cross-check against below.

# %%
# real-matrix generators (basis order: S, J, P1, P2)
S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)  # uniform scale
J = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)  # rotation
P1 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)  # x-translation
P2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)  # y-translation
BASIS = [S, J, P1, P2]


def alg_real(alpha, beta):
    """Algebra element X(α, β) as a real 3x3 matrix."""
    return alpha.real * S + alpha.imag * J + beta.real * P1 + beta.imag * P2


def alg_coords(X):
    """Coordinates (λ, ω, uₓ, u_y) of an algebra matrix in the basis {S, J, P1, P2}."""
    return np.array([X[0, 0], X[1, 0], X[0, 2], X[1, 2]])


def V(alpha):
    """Complex 'Jacobian' (e^α − 1)/α, with the removable limit V(0) = 1."""
    return 1.0 + 0j if abs(alpha) < 1e-12 else (np.exp(alpha) - 1) / alpha


def exp_sim2(alpha, beta):
    """Closed-form exp on sim(2): returns the real 3x3 group element (a=e^α, b=V(α)β)."""
    return sim2_real(np.exp(alpha), V(alpha) * beta)


for alpha, beta, desc in [
    (0.0 + 1.2j, 0 + 0j, "pure rotation"),
    (0.4 + 0.0j, 0 + 0j, "pure scaling"),
    (0.0 + 0.0j, 1.0 + 0.5j, "pure translation"),
    (0.25 + 1.0j, 0.6 - 0.3j, "spiral + translation"),
]:
    closed = exp_sim2(alpha, beta)
    series = expm(alg_real(alpha, beta))
    print(f"{desc:22s}  closed form (complex V) == expm: {np.allclose(closed, series)}")

# --- Visualize logarithmic spirals: integral curves exp(tX) of a point ---
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.set_title(
    "Integral curves exp(t·X) on SIM(2): scale+rotation ⇒ logarithmic spirals",
    color="#ccc",
    fontsize=10,
)
ax.set_aspect("equal")
ax.grid(True)
ax.axhline(0, color="#555", lw=0.8)
ax.axvline(0, color="#555", lw=0.8)
ts = np.linspace(0, 6, 300)
z_start = 1.0 + 0j
for alpha, color, label in [
    (-0.25 + 1.0j, BLUE, "λ<0: inward spiral"),
    (0.0 + 1.0j, GREEN, "λ=0: circle (SE(2)-like)"),
    (0.15 + 1.0j, ORANGE, "λ>0: outward spiral"),
]:
    pts = np.array([apply_real(exp_sim2(alpha * t, 0 + 0j), z_start) for t in ts])
    ax.plot(pts.real, pts.imag, "-", color=color, lw=2, label=label)
ax.plot(z_start.real, z_start.imag, "o", color="white", ms=8, zorder=5)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. The ad Representation: the Bracket is Complex Multiplication
#
# The Lie bracket of two algebra elements reduces, in complex form, to
#
# $$[(\alpha_1, \beta_1), (\alpha_2, \beta_2)] = \big(0,\; \alpha_1\beta_2 - \alpha_2\beta_1\big).$$
#
# The scale-rotation part $\alpha$ is **abelian** (its bracket vanishes — scalings and rotations
# commute), while the translation part transforms by **complex multiplication** by $\alpha$. So
# $\text{ad}_{(\alpha_0,\beta_0)}(\alpha,\beta) = (0,\ \alpha_0\beta - \alpha\beta_0)$; as a
# $4\times4$ matrix it is built from these. We confirm the matrix commutator agrees.


# %%
def bracket(A, B):
    return A @ B - B @ A


def ad_matrix(alpha0, beta0):
    """ad_X as a 4x4 matrix (basis S,J,P1,P2), built from the matrix commutator."""
    X = alg_real(alpha0, beta0)
    return np.column_stack([alg_coords(bracket(X, E)) for E in BASIS])


def ad_complex(alpha0, beta0):
    """ad_X as a 4x4 matrix via the complex law (α,β) ↦ (0, α0·β − α·β0)."""
    cols = []
    for alpha, beta in [(1 + 0j, 0j), (1j, 0j), (0j, 1 + 0j), (0j, 1j)]:
        bp = alpha0 * beta - alpha * beta0  # α' = 0 always
        cols.append([0.0, 0.0, bp.real, bp.imag])
    return np.column_stack(cols)


a0, b0 = 0.3 + 0.9j, 0.7 - 0.4j
print(
    "ad from matrix commutator == ad from complex law (0, α0β−αβ0):",
    np.allclose(ad_matrix(a0, b0), ad_complex(a0, b0)),
)
# the bracket itself, both ways, for two elements
X1, X2 = alg_real(a0, b0), alg_real(-0.2 + 0.5j, 1.1 + 0.2j)
brk_complex = 0, (a0) * (1.1 + 0.2j) - (-0.2 + 0.5j) * (b0)  # (0, α1β2 − α2β1)
print(
    "matrix [X1,X2] == complex (0, α1β2−α2β1):",
    np.allclose(
        alg_coords(bracket(X1, X2)), [0, 0, brk_complex[1].real, brk_complex[1].imag]
    ),
)
print("\nad_X (4×4) =\n", np.round(ad_matrix(a0, b0), 3))

# %% [markdown]
# ## 5. The Ad Representation: Conjugation as Complex Arithmetic
#
# For $g = (a, b)$, conjugation acts on an algebra element by
#
# $$\text{Ad}_{(a,b)}(\alpha, \beta) = \big(\alpha,\; a\,\beta - \alpha\, b\big).$$
#
# The scale-rotation generator $\alpha$ is **invariant** (the $\mathbb{C}^\times$ factor is
# abelian), while the translation generator is scale-rotated by $a$ and coupled to $\alpha$ through
# $-\alpha b$ — the SIM(2) echo of SE(2)'s adjoint. We verify it against matrix conjugation.


# %%
def Ad_matrix(a, b):
    """Ad_g as a 4x4 matrix (basis S,J,P1,P2), built from conjugation g·E·g⁻¹."""
    g = sim2_real(a, b)
    g_inv = sim2_real(*sim2_inverse(a, b))
    return np.column_stack([alg_coords(g @ E @ g_inv) for E in BASIS])


def Ad_complex(a, b):
    """Ad_g as a 4x4 matrix via the complex law (α,β) ↦ (α, aβ − αb)."""
    cols = []
    for alpha, beta in [(1 + 0j, 0j), (1j, 0j), (0j, 1 + 0j), (0j, 1j)]:
        ap = alpha
        bp = a * beta - alpha * b
        cols.append([ap.real, ap.imag, bp.real, bp.imag])
    return np.column_stack(cols)


a, b = 1.3 * np.exp(1j * 0.7), 0.6 - 0.5j
print(
    "Ad from conjugation == Ad from complex law (α, aβ−αb):",
    np.allclose(Ad_matrix(a, b), Ad_complex(a, b)),
)
print("\nAd_g (4×4) =\n", np.round(Ad_matrix(a, b), 3))

# Ad really conjugates: g·exp(Y)·g⁻¹ = exp(Ad_g·Y)
ya, yb = 0.2 + 0.6j, 0.4 - 0.1j
g = sim2_real(a, b)
g_inv = sim2_real(*sim2_inverse(a, b))
lhs = g @ exp_sim2(ya, yb) @ g_inv
Y_coords = Ad_matrix(a, b) @ alg_coords(alg_real(ya, yb))
rhs = exp_sim2(Y_coords[0] + 1j * Y_coords[1], Y_coords[2] + 1j * Y_coords[3])
print("g·exp(Y)·g⁻¹ == exp(Ad_g·Y):", np.allclose(lhs, rhs))

# %% [markdown]
# ## 6. The Key Relationship: $\text{Ad}_{e^X} = e^{\text{ad}_X}$
#
# $$\boxed{\text{Ad}_{e^{X}} = e^{\text{ad}_{X}}}$$
#
# The $4\times4$ adjoint of the group element $e^{X}$ equals the matrix exponential of the
# $4\times4$ operator $\text{ad}_{X}$. We verify numerically across the four motion types.

# %%
print("Verifying  Ad_{exp(X)} = exp(ad_X)  (4×4 matrices):")
print("=" * 54)
for alpha, beta, desc in [
    (0.0 + 1.2j, 0 + 0j, "pure rotation"),
    (0.5 + 0.0j, 0 + 0j, "pure scaling"),
    (0.0 + 0.0j, 1.0 + 0.5j, "pure translation"),
    (0.25 + 1.0j, 0.6 - 0.3j, "spiral + translation"),
]:
    g = exp_sim2(alpha, beta)  # real 3x3 group element
    a_g, b_g = g[0, 0] + 1j * g[1, 0], g[0, 2] + 1j * g[1, 2]
    LHS = Ad_matrix(a_g, b_g)
    RHS = expm(ad_matrix(alpha, beta))
    err = np.linalg.norm(LHS - RHS)
    print(
        f"  {desc:22s}:  ||Ad_exp − exp_ad||_F = {err:.2e}  →  match: {np.allclose(LHS, RHS)}"
    )

# %% [markdown]
# ## Summary
#
# | | meaning | complex form |
# |---|---|---|
# | Group SIM(2) | similarity $z \mapsto az + b$ | $(a, b),\ a\in\mathbb{C}^\times,\ b\in\mathbb{C}$ |
# | composition | rotate-scale-translate | $(a_1a_2,\ a_1b_2 + b_1)$ |
# | algebra $\mathfrak{sim}(2)$ | scale + rot + 2 translations | $(\alpha, \beta),\ \alpha,\beta\in\mathbb{C}$ |
# | exp | spiral motion | $a = e^\alpha,\ b = \frac{e^\alpha - 1}{\alpha}\beta$ |
# | bracket $\text{ad}$ | $[(\alpha_1,\beta_1),(\alpha_2,\beta_2)]$ | $(0,\ \alpha_1\beta_2 - \alpha_2\beta_1)$ |
# | adjoint $\text{Ad}$ | $\text{Ad}_{(a,b)}(\alpha,\beta)$ | $(\alpha,\ a\beta - \alpha b)$ |
#
# **The big picture — a family of "complex" rotation groups:**
# - **SO(2) $\cong$ U(1)**: unit complex numbers $e^{i\theta}$ — pure rotation (abelian).
# - **$\mathbb{C}^\times$**: all nonzero complex numbers $e^{\lambda + i\omega}$ — rotation *and* scale.
# - **SIM(2) $= \mathbb{C}^\times \ltimes \mathbb{C}$**: scale-rotation *and* translation, $z \mapsto az + b$.
#
# SE(2) is the $|a| = 1$ subgroup of SIM(2) (drop scaling); SO(2) is the further $b = 0$ subgroup.
# Adding the scale degree of freedom turns SE(2)'s circular orbits into **logarithmic spirals**,
# and — because $a$ ranges over all of $\mathbb{C}^\times$ — lets the *entire* Lie theory of the
# group be written in the language of complex numbers.
