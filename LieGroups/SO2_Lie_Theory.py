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
# # Lie Groups and Lie Algebras: SO(2) Illustrated
#
# **SO(2)** is the special orthogonal group in 2D — the group of rotations of the plane
# about the origin. It is the simplest non-trivial Lie group: a 1-dimensional, **abelian**
# (commutative) group whose manifold is the circle $S^1$.
#
# Because it is abelian, SO(2) is the *baseline* case where all the Lie-theoretic machinery
# collapses to something trivial:
# 1. The group structure of SO(2) (rotation = angle addition)
# 2. The Lie algebra $\mathfrak{so}(2)$ — a 1D line of skew-symmetric matrices
# 3. The Lie bracket (**ad**) — identically **zero** (abelian!)
# 4. The adjoint representation (**Ad**) — the **identity** (abelian!)
# 5. The relationship $\text{Ad}_{e^X} = e^{\text{ad}_X}$ — both sides equal 1
# 6. The isomorphism $SO(2) \cong U(1)$ — rotations as the unit complex numbers under multiplication
#
# Contrast this everywhere with the non-commutative SE(2)/SO(3)/SE(3) notebooks: there the
# bracket and adjoints are where all the interesting geometry lives.

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
# ## 1. SO(2): The Group
#
# A group element is a $2\times 2$ rotation matrix:
#
# $$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$
#
# Composition is simply **angle addition**: $R(\alpha)R(\beta) = R(\alpha + \beta) = R(\beta)R(\alpha)$.
# The group is therefore **abelian**, and its manifold is the circle $S^1$ (angles mod $2\pi$).


# %%
def rotation_matrix(theta):
    """2x2 rotation matrix for angle theta (radians) — an element of SO(2)."""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


# --- Verify the group is abelian and composition = angle addition ---
a, b = 0.7, 1.3
print("R(a)R(b) =\n", np.round(rotation_matrix(a) @ rotation_matrix(b), 4))
print("\nR(a+b) =\n", np.round(rotation_matrix(a + b), 4))
print("\nR(b)R(a) =\n", np.round(rotation_matrix(b) @ rotation_matrix(a), 4))
print(
    "\nAbelian? R(a)R(b) == R(b)R(a):",
    np.allclose(
        rotation_matrix(a) @ rotation_matrix(b), rotation_matrix(b) @ rotation_matrix(a)
    ),
)
print(
    "Composition == angle addition:",
    np.allclose(rotation_matrix(a) @ rotation_matrix(b), rotation_matrix(a + b)),
)

# --- Visualize the group action and the group manifold (the circle) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("SO(2): Rotations of the Plane", fontsize=14, color="white", y=1.0)

# A simple asymmetric shape so orientation is visible
shape = np.array([[0.0, 0.6, 0.3, 0.0], [0.0, 0.0, 0.5, 0.0]])

ax = axes[0]
ax.set_title(
    "Group action: R(θ) rotates a shape about the origin", color="#ccc", fontsize=10
)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
ax.grid(True)
thetas = np.linspace(0, 2 * np.pi, 9)[:-1]
colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))
for theta, c in zip(thetas, colors):
    t = rotation_matrix(theta) @ shape
    ax.fill(t[0], t[1], alpha=0.25, color=c)
    ax.plot(t[0], t[1], "-", color=c, lw=1.5)
ax.plot(0, 0, "o", color="white", ms=7, zorder=5)

ax = axes[1]
ax.set_title("Group manifold: SO(2) ≅ the circle S¹", color="#ccc", fontsize=10)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal")
ax.grid(True)
phi = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(phi), np.sin(phi), "-", color=BLUE, lw=2)
for theta, label in [(0, "θ=0  (e)"), (np.pi / 2, "θ=π/2"), (2.5, "θ=2.5")]:
    ax.plot(np.cos(theta), np.sin(theta), "o", color=ORANGE, ms=9, zorder=5)
    ax.annotate(
        label,
        xy=(np.cos(theta), np.sin(theta)),
        xytext=(1.25 * np.cos(theta), 1.25 * np.sin(theta)),
        color="#ccc",
        fontsize=9,
        ha="center",
    )
ax.text(0, 0, "each point = one\nrotation R(θ)", color="#888", fontsize=8, ha="center")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The Lie Algebra $\mathfrak{so}(2)$: Tangent Space at Identity
#
# Differentiating $R(\theta)$ at $\theta = 0$ gives the single **generator**:
#
# $$G = \left.\frac{dR}{d\theta}\right|_{0} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$
#
# The Lie algebra $\mathfrak{so}(2) = \{\theta G : \theta \in \mathbb{R}\}$ is the 1D space of
# $2\times 2$ skew-symmetric matrices. The exponential map recovers the group via the
# 2D Rodrigues formula $e^{\theta G} = \cos\theta\, I + \sin\theta\, G = R(\theta)$.
#
# > **Note — closed forms are a luxury, not a guarantee.** SO(2), SO(3), SE(2), SE(3) all admit
# > tidy closed-form exponentials (here, Rodrigues) because they are low-dimensional matrix groups
# > whose generators satisfy a simple polynomial relation ($G^2 = -I$). For a general Lie algebra
# > there is *no* such closed form, and one falls back to the defining matrix-exponential series
# > $e^{X} = \sum_k X^k / k!$ — which is exactly what `scipy.linalg.expm` evaluates below.

# %%
G = np.array([[0.0, -1.0], [1.0, 0.0]])  # the single generator of so(2)


def algebra_element(theta):
    """General element of so(2): theta * G (a skew-symmetric 2x2 matrix)."""
    return theta * G


print("Generator G of so(2):")
print(G)
print("\nG is skew-symmetric (Gᵀ = −G):", np.allclose(G.T, -G))
print("G² = −I:", np.allclose(G @ G, -np.eye(2)))

# exp(theta G) should equal R(theta), via Rodrigues and via scipy.expm
theta = 0.9
rodrigues = np.cos(theta) * np.eye(2) + np.sin(theta) * G
print(f"\nθ = {theta}")
print("exp(θG) via Rodrigues  cosθ·I + sinθ·G =\n", np.round(rodrigues, 4))
print("exp(θG) via scipy.expm =\n", np.round(expm(theta * G), 4))
print("R(θ) directly =\n", np.round(rotation_matrix(theta), 4))
print(
    "\nAll three agree:",
    np.allclose(rodrigues, expm(theta * G))
    and np.allclose(rodrigues, rotation_matrix(theta)),
)

# --- Visualize the exponential map: the real line wraps onto the circle ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "exp: 𝖘𝖔(2) → SO(2)  —  the real line θ wraps onto the circle (2π-periodic)",
    color="white",
    fontsize=12,
)

ax = axes[0]
ax.set_title(
    "Lie algebra 𝖘𝖔(2) ≅ ℝ  (tangent line at identity)", color="#ccc", fontsize=10
)
ax.set_xlim(-7, 7)
ax.set_ylim(-1, 1)
ax.axhline(0, color=ORANGE, lw=2)
for theta in [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]:
    ax.plot(theta, 0, "o", color=ORANGE, ms=8)
ax.set_yticks([])
ax.set_xlabel("θ  (coordinate along the generator G)")
ax.text(0, 0.3, "θ=0 ↦ identity e", color="#ccc", fontsize=9, ha="center")

ax = axes[1]
ax.set_title("Image in SO(2): R(θ) = exp(θG)", color="#ccc", fontsize=10)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.grid(True)
phi = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(phi), np.sin(phi), "-", color=BLUE, lw=2)
for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    ax.plot(np.cos(theta), np.sin(theta), "o", color=ORANGE, ms=6)
ax.plot(1, 0, "o", color="white", ms=8, zorder=5)
ax.text(1.05, 0.05, "e", color="white", fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The ad Representation: the Bracket Vanishes
#
# For $X \in \mathfrak{so}(2)$, $\text{ad}_X(Y) = [X, Y] = XY - YX$. But $\mathfrak{so}(2)$ is
# 1-dimensional, spanned by $G$, and $G$ commutes with itself, so **every bracket is zero**:
#
# $$[X, Y] = 0 \quad \forall\, X, Y \in \mathfrak{so}(2) \qquad\Longrightarrow\qquad \text{ad}_X = 0$$
#
# As a $1\times 1$ matrix in the basis $\{G\}$, $\text{ad}_X$ is the zero map. This is the
# defining feature of an **abelian** Lie algebra.


# %%
def lie_bracket(A, B):
    """Lie bracket as the matrix commutator [A, B] = AB − BA."""
    return A @ B - B @ A


X = algebra_element(1.7)
Y = algebra_element(-0.4)
print("X = 1.7·G,   Y = −0.4·G")
print("\n[X, Y] = XY − YX =\n", lie_bracket(X, Y))
print("\nBracket is identically zero → so(2) is ABELIAN.")
print("Therefore ad_X = 0 (the 1×1 zero matrix) for every X.")

# %% [markdown]
# ## 4. The Ad Representation: the Identity Map
#
# For $g = R(\phi) \in SO(2)$, the adjoint is $\text{Ad}_g(Y) = g\, Y\, g^{-1}$. Because the
# group is abelian, $g$ and $Y$ commute, so
#
# $$\text{Ad}_g(Y) = g\, Y\, g^{-1} = Y\, g\, g^{-1} = Y$$
#
# for **every** $g$. As a $1\times 1$ matrix in the basis $\{G\}$, $\text{Ad}_g = (1)$ — the
# identity, completely independent of $g$.


# %%
def Ad_scalar(g):
    """Ad_g as a 1x1 matrix in the basis {G}: coordinate of g G g^{-1} along G."""
    conj = g @ G @ np.linalg.inv(g)
    return conj[
        1, 0
    ]  # coordinate of a skew matrix [[0,-c],[c,0]] along G is its (1,0) entry


for phi in [0.0, 0.5, np.pi / 2, 2.7]:
    g = rotation_matrix(phi)
    print(f"Ad_{{R({phi:.3f})}} = {Ad_scalar(g):.6f}   (always 1, independent of φ)")

print("\nAd_g ≡ 1 for all g → conjugation does nothing in an abelian group.")

# %% [markdown]
# ## 5. The Relationship $\text{Ad}_{e^X} = e^{\text{ad}_X}$
#
# The fundamental theorem connecting group and algebra adjoints still holds — trivially:
#
# $$\text{Ad}_{e^X} = (1) \qquad\text{and}\qquad e^{\text{ad}_X} = e^{0} = (1)$$
#
# Both sides are the $1\times 1$ identity for every $X$. SO(2) satisfies the theorem in the
# most degenerate possible way; the non-commutative groups are where it carries real content.

# %%
print("Verifying  Ad_{exp(X)} = exp(ad_X)  for several X in so(2):")
print("=" * 52)
for theta in [0.3, 1.2, -2.0, np.pi]:
    X = algebra_element(theta)
    LHS = Ad_scalar(expm(X))  # Ad of the group element exp(X)
    ad_X = 0.0  # ad_X is the 1x1 zero map
    RHS = np.exp(ad_X)  # exp(0) = 1
    print(
        f"  X = {theta:+.3f}·G :  Ad_exp(X) = {LHS:.6f},  exp(ad_X) = {RHS:.6f}  →  match: {np.isclose(LHS, RHS)}"
    )

# --- Visualize abelian commutativity: the SE(2) "bracket gap" closes to zero here ---
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.set_title(
    "Abelian: R(α)·R(β) and R(β)·R(α) land on the SAME rotation\n(the non-commutativity 'gap' of SE(2)/SO(3) is zero here)",
    color="#ccc",
    fontsize=10,
)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.grid(True)
phi = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(phi), np.sin(phi), "-", color="#444", lw=1.5)

alpha, beta = 0.6, 1.4
p_marker = np.array([1.0, 0.0])  # track where the "1 o'clock" reference point goes
# path α then β
path_ab = [rotation_matrix(t) @ p_marker for t in np.linspace(0, alpha, 30)]
path_ab += [rotation_matrix(alpha + t) @ p_marker for t in np.linspace(0, beta, 30)]
path_ab = np.array(path_ab)
# path β then α
path_ba = [rotation_matrix(t) @ p_marker for t in np.linspace(0, beta, 30)]
path_ba += [rotation_matrix(beta + t) @ p_marker for t in np.linspace(0, alpha, 30)]
path_ba = np.array(path_ba)

ax.plot(
    path_ab[:, 0],
    path_ab[:, 1],
    "-",
    color=BLUE,
    lw=3,
    alpha=0.7,
    label="R(α) then R(β)",
)
ax.plot(
    path_ba[:, 0], path_ba[:, 1], "--", color=ORANGE, lw=2.5, label="R(β) then R(α)"
)
end = rotation_matrix(alpha + beta) @ p_marker
ax.plot(*end, "*", color=GREEN, ms=20, zorder=6, label="common endpoint")
ax.plot(*p_marker, "o", color="white", ms=8, zorder=5)
ax.text(1.02, 0.05, "start", color="white", fontsize=9)
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. SO(2) ≅ U(1): the Unit Complex Numbers
#
# SO(2) is **isomorphic** to **U(1)**, the group of unit-modulus complex numbers
# $\{z \in \mathbb{C} : |z| = 1\}$ under multiplication. The isomorphism is
#
# $$\varphi : SO(2) \to U(1), \qquad R(\theta) \longmapsto e^{i\theta} = \cos\theta + i\sin\theta,$$
#
# and it respects the group operation: composing rotations corresponds to **multiplying** complex
# numbers, because both amount to **adding angles**:
#
# $$\varphi\big(R(\alpha)R(\beta)\big) = e^{i(\alpha+\beta)} = e^{i\alpha}\,e^{i\beta}
#   = \varphi\big(R(\alpha)\big)\,\varphi\big(R(\beta)\big).$$
#
# Concretely, identifying a plane vector $(x, y)$ with the complex number $x + iy$, **rotating by
# $R(\theta)$ is the same as multiplying by $e^{i\theta}$**. The isomorphism descends to the Lie
# algebras as well: the generator $G$ (which satisfies $G^2 = -I$) corresponds to the imaginary
# unit $i$ (which satisfies $i^2 = -1$), so $e^{\theta G} = R(\theta)$ mirrors Euler's formula
# $e^{i\theta} = \cos\theta + i\sin\theta$. SO(2), U(1), and the circle $S^1$ are three faces of
# the same group: the canonical 1-dimensional compact abelian Lie group.


# %%
def so2_to_complex(g):
    """Map an SO(2) matrix R(θ) to the unit complex number e^{iθ} = R[0,0] + i·R[1,0]."""
    return g[0, 0] + 1j * g[1, 0]


# Group homomorphism: matrix product ↔ complex product
a, b = 0.7, 1.3
za, zb = so2_to_complex(rotation_matrix(a)), so2_to_complex(rotation_matrix(b))
print("φ(R(a)) = e^{ia} =", np.round(za, 4), " (|z| =", round(abs(za), 6), ")")
print("φ(R(b)) = e^{ib} =", np.round(zb, 4))
print("\nHomomorphism  φ(R(a)·R(b)) == φ(R(a))·φ(R(b)):")
print(
    "  φ(R(a)R(b)) =",
    np.round(so2_to_complex(rotation_matrix(a) @ rotation_matrix(b)), 6),
)
print("  e^{ia}·e^{ib} =", np.round(za * zb, 6))
print(
    "  match:",
    np.isclose(so2_to_complex(rotation_matrix(a) @ rotation_matrix(b)), za * zb),
)

# Rotating a vector ↔ multiplying by a unit complex number
v = np.array([0.8, 0.3])
theta = 1.1
rotated_vec = rotation_matrix(theta) @ v  # rotate in R^2
rotated_cpx = np.exp(1j * theta) * (v[0] + 1j * v[1])  # multiply in C
print("\nR(θ)·(x,y)  vs  e^{iθ}·(x+iy):")
print("  R(θ)·v       =", np.round(rotated_vec, 4))
print("  e^{iθ}·(x+iy) =", np.round([rotated_cpx.real, rotated_cpx.imag], 4))
print("  match:", np.allclose(rotated_vec, [rotated_cpx.real, rotated_cpx.imag]))

# Algebra-level correspondence: G ↔ i  (both square to −1)
print(
    "\nAlgebra: G² = −I  ↔  i² = −1  →  exp(θG) = R(θ)  mirrors  e^{iθ} = cosθ + i sinθ"
)

# --- Visualize the isomorphism: the Argand plane ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "SO(2) ≅ U(1):  rotating vectors  ⟷  multiplying unit complex numbers",
    color="white",
    fontsize=12,
)

# Left: a rotation acting on a vector in R^2
ax = axes[0]
ax.set_title("In ℝ²:  v ↦ R(θ)·v", color="#ccc", fontsize=10)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.grid(True)
ax.axhline(0, color="#555", lw=0.8)
ax.axvline(0, color="#555", lw=0.8)
ax.annotate(
    "", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.5)
)
ax.text(v[0] + 0.05, v[1], "v", color=BLUE, fontsize=12)
ax.annotate(
    "",
    xy=rotated_vec,
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5),
)
ax.text(rotated_vec[0] + 0.05, rotated_vec[1], "R(θ)·v", color=ORANGE, fontsize=10)

# Right: the same as a unit complex number on the unit circle (Argand plane)
ax = axes[1]
ax.set_title(
    "In ℂ:  z ↦ e^{iθ}·z   (e^{iθ} lives on the unit circle U(1))",
    color="#ccc",
    fontsize=10,
)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.grid(True)
ax.axhline(0, color="#555", lw=0.8)
ax.axvline(0, color="#555", lw=0.8)
phi = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(phi), np.sin(phi), "-", color="#444", lw=1.5)
z = v[0] + 1j * v[1]
ax.annotate(
    "",
    xy=(z.real, z.imag),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.5),
)
ax.text(z.real + 0.05, z.imag, "z = x+iy", color=BLUE, fontsize=11)
ax.annotate(
    "",
    xy=(rotated_cpx.real, rotated_cpx.imag),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5),
)
ax.text(
    rotated_cpx.real + 0.05, rotated_cpx.imag, "e^{iθ}·z", color=ORANGE, fontsize=10
)
# show e^{iθ} itself as the multiplier on the unit circle
ax.plot(np.cos(theta), np.sin(theta), "o", color=GREEN, ms=9, zorder=5)
ax.annotate(
    "e^{iθ}",
    xy=(np.cos(theta), np.sin(theta)),
    xytext=(np.cos(theta) - 0.1, np.sin(theta) + 0.15),
    color=GREEN,
    fontsize=10,
)
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | | **ad** | **Ad** |
# |---|---|---|
# | Domain | Lie algebra $\mathfrak{so}(2)$ (≅ ℝ) | Lie group SO(2) (≅ S¹) |
# | Formula | $\text{ad}_X(Y) = [X,Y]$ | $\text{Ad}_g(Y) = gYg^{-1}$ |
# | Value | $0$ (bracket vanishes) | $1$ (identity, ∀g) |
# | Connection | $\text{Ad}_{e^X} = e^{\text{ad}_X}$ | $(1) = e^{0} = (1)$ |
#
# **Key takeaway:** SO(2) is *abelian*, so it is the trivial end of the Lie-theory spectrum:
# - The exponential map $\theta \mapsto R(\theta)$ wraps the line $\mathbb{R}$ onto the circle $S^1$.
# - The Lie bracket is identically zero, so $\text{ad}_X = 0$.
# - Conjugation does nothing, so $\text{Ad}_g = 1$ for every $g$.
#
# Everything that makes SE(2), SO(3), and SE(3) interesting — non-commutativity, a non-trivial
# bracket, frame-dependent adjoints — is exactly what SO(2) *lacks*. It is the control case.
