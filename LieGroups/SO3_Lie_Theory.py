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
# # Lie Groups and Lie Algebras: SO(3) Illustrated
#
# **SO(3)** is the special orthogonal group in 3D — the group of rotations of 3D space.
# Unlike SO(2) it is **non-abelian** (rotations about different axes do not commute), which
# makes it the first genuinely rich rotation group and the workhorse of 3D robotics, graphics,
# and computer vision.
#
# This notebook illustrates:
# 1. The group structure of SO(3) (3×3 orthogonal matrices, $\det = +1$)
# 2. The Lie algebra $\mathfrak{so}(3)$ — skew-symmetric matrices, the **hat** map $\wedge$
# 3. The Lie bracket (**ad**) — equals the **cross product**: $[\hat a, \hat b] = \widehat{a\times b}$
# 4. The adjoint representation (**Ad**) — with the beautiful identity $\text{Ad}_R = R$
# 5. The relationship $\text{Ad}_{e^X} = e^{\text{ad}_X}$
# 6. Geometric visualizations in 3D

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
# ## 1. SO(3): The Group
#
# A group element is a $3\times3$ matrix $R$ with $R^\top R = I$ and $\det R = +1$. It rotates
# vectors of $\mathbb{R}^3$ while preserving lengths, angles, and handedness. Composition is
# matrix multiplication, and — crucially — it is **non-commutative**.


# %%
def rotation_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rotation_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rotation_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def draw_frame_3d(ax, R, origin=(0, 0, 0), scale=1.0, alpha=1.0, lw=2.5, labels=False):
    """Draw a coordinate frame (columns of R) at `origin`."""
    o = np.asarray(origin, dtype=float)
    for i, (col, label) in enumerate(zip([RED, GREEN, BLUE], ["x", "y", "z"])):
        axis = R[:, i] * scale
        ax.quiver(*o, *axis, color=col, lw=lw, alpha=alpha, arrow_length_ratio=0.15)
        if labels:
            ax.text(*(o + axis * 1.12), label, color=col, fontsize=10)


Ra = rotation_z(np.pi / 4) @ rotation_x(np.pi / 6)
Rb = rotation_y(np.pi / 3)
print("Composition is non-commutative:  Ra·Rb ≠ Rb·Ra")
print("Ra·Rb =\n", np.round(Ra @ Rb, 3))
print("\nRb·Ra =\n", np.round(Rb @ Ra, 3))
print(
    "\n||Ra·Rb − Rb·Ra||_F =", round(np.linalg.norm(Ra @ Rb - Rb @ Ra), 4), "(nonzero!)"
)

# --- Visualize: a point cloud (an 'L' shaped object) rotated by R ---
fig = plt.figure(figsize=(13, 5.5))
fig.suptitle("SO(3): Rotations of 3D Space", color="white", fontsize=14)

# little 3D 'L' tetromino-ish object made of points
obj = np.array(
    [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
).T  # 3 x N

for idx, (R, title) in enumerate(
    [
        (np.eye(3), "Identity e"),
        (rotation_z(0.9) @ rotation_x(0.6), "Rotated by R = Rz(0.9)·Rx(0.6)"),
    ]
):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    p = R @ obj
    ax.scatter(p[0], p[1], p[2], color=ORANGE, s=40)
    for k in range(p.shape[1]):
        ax.plot([0, p[0, k]], [0, p[1, k]], [0, p[2, k]], color=ORANGE, alpha=0.3, lw=1)
    draw_frame_3d(ax, R, scale=1.5, labels=True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect((1, 1, 1))

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The Lie Algebra $\mathfrak{so}(3)$: the hat map
#
# The Lie algebra $\mathfrak{so}(3)$ is the space of **skew-symmetric** $3\times3$ matrices
# ($X^\top = -X$). It is 3-dimensional, and the **hat** map $\wedge:\mathbb{R}^3\to\mathfrak{so}(3)$
# identifies a vector $\omega = (\omega_x,\omega_y,\omega_z)$ with the matrix that implements the
# cross product, $\hat\omega\, v = \omega \times v$:
#
# $$\hat\omega = \begin{pmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{pmatrix}$$
#
# The three **generators** $G_i = \hat e_i$ are the infinitesimal rotations about the x, y, z axes.


# %%
def hat(w):
    """Map a 3-vector to its skew-symmetric matrix (an element of so(3))."""
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def vee(W):
    """Inverse of hat: extract the 3-vector from a skew-symmetric matrix."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


G1, G2, G3 = hat([1, 0, 0]), hat([0, 1, 0]), hat([0, 0, 1])

print("Generators of so(3) (Gᵢ = ê_i):")
for name, Gi in [("G1 (about x)", G1), ("G2 (about y)", G2), ("G3 (about z)", G3)]:
    print(f"\n{name}:\n{Gi.astype(int)}")

# sanity: hat(w) implements the cross product
w, v = np.array([0.2, -1.0, 0.5]), np.array([1.0, 2.0, -0.5])
print("\nhat(w)·v == w × v :", np.allclose(hat(w) @ v, np.cross(w, v)))
print("hat(w) is skew-symmetric:", np.allclose(hat(w).T, -hat(w)))

# %% [markdown]
# ## The exponential map: Rodrigues' rotation formula
#
# For $\hat\omega \in \mathfrak{so}(3)$ with angle $\theta = \|\omega\|$ and unit axis
# $\hat n = \omega/\theta$, the exponential has the celebrated closed form
#
# $$e^{\hat\omega} = I + \frac{\sin\theta}{\theta}\,\hat\omega + \frac{1-\cos\theta}{\theta^2}\,\hat\omega^2$$
#
# (rotation by $\theta$ about axis $n$). This is **Rodrigues' formula**.
#
# > **Caveat — not every Lie algebra is this lucky.** Rodrigues exists because $\hat\omega$
# > satisfies $\hat\omega^3 = -\theta^2\hat\omega$, which truncates the exponential series to three
# > terms. Most Lie algebras have no such closed form; you would simply sum the series via
# > `scipy.linalg.expm`. We cross-check Rodrigues against `expm` below precisely to make the point
# > that the closed form is an *optimization*, not a different object.


# %%
def exp_so3(w):
    """Rodrigues' formula: exp(hat(w)) as a 3x3 rotation matrix."""
    theta = np.linalg.norm(w)
    W = hat(w)
    if theta < 1e-12:
        return np.eye(3) + W  # first-order limit, exact to O(theta^2)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1 - np.cos(theta)) / theta**2) * (W @ W)
    )


for w in [
    np.array([0, 0, np.pi / 2]),
    np.array([0.3, -0.7, 1.1]),
    np.array([1.0, 1.0, 1.0]),
]:
    R_rod = exp_so3(w)
    R_expm = expm(hat(w))
    ok = np.allclose(R_rod, R_expm)
    orth = np.allclose(R_rod.T @ R_rod, np.eye(3)) and np.isclose(
        np.linalg.det(R_rod), 1.0
    )
    print(
        f"w = {np.round(w, 3)}:  Rodrigues == expm: {ok},  is a valid rotation: {orth}"
    )

# --- Visualize integral curves exp(t·Gi): a point traces a circle on the sphere ---
fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(
    "Integral curves exp(t·ω̂):  one-parameter rotation subgroups",
    color="#ccc",
    fontsize=11,
)
ax.set_facecolor("#1a1a2e")
# faint sphere
u, vv = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
ax.plot_surface(
    np.cos(u) * np.sin(vv), np.sin(u) * np.sin(vv), np.cos(vv), color="#333", alpha=0.15
)
p0 = np.array([0.0, 0.0, 1.0])  # north pole, tracked point
ts = np.linspace(0, 2 * np.pi, 80)
for axis, color, label in [
    (np.array([1.0, 0, 0]), BLUE, "about x"),
    (np.array([0, 1.0, 0]), ORANGE, "about y"),
    (np.array([1.0, 1.0, 0.4]), GREEN, "about (1,1,0.4)"),
]:
    axis = axis / np.linalg.norm(axis)
    pts = np.array([exp_so3(t * axis) @ p0 for t in ts])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=2.5, label=label)
ax.scatter(*p0, color="white", s=50, zorder=5)
ax.set_box_aspect((1, 1, 1))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The ad Representation: the bracket IS the cross product
#
# For $\hat a, \hat b \in \mathfrak{so}(3)$, the Lie bracket equals the hat of the cross product:
#
# $$[\hat a, \hat b] = \hat a\,\hat b - \hat b\,\hat a = \widehat{a \times b}$$
#
# So $\text{ad}_{\hat\omega}$, as a $3\times3$ matrix acting on $\mathbb{R}^3$ (in the basis
# $\{G_1,G_2,G_3\}$), is simply $\hat\omega$ itself. The structure constants are the
# Levi–Civita symbol: $[G_i, G_j] = \varepsilon_{ijk} G_k$.


# %%
def lie_bracket(A, B):
    return A @ B - B @ A


# bracket == hat of cross product
a, b = np.array([0.5, -0.2, 1.0]), np.array([-1.0, 0.3, 0.7])
print(
    "[â, b̂] == hat(a × b):",
    np.allclose(lie_bracket(hat(a), hat(b)), hat(np.cross(a, b))),
)

# structure constants: [G1,G2] = G3, [G2,G3] = G1, [G3,G1] = G2
print("[G1, G2] == G3:", np.allclose(lie_bracket(G1, G2), G3))
print("[G2, G3] == G1:", np.allclose(lie_bracket(G2, G3), G1))
print("[G3, G1] == G2:", np.allclose(lie_bracket(G3, G1), G2))


def ad_matrix(w):
    """ad_{hat(w)} as a 3x3 matrix in the basis {G1,G2,G3}: column j = vee([hat(w), Gj])."""
    basis = [G1, G2, G3]
    return np.column_stack([vee(lie_bracket(hat(w), Gj)) for Gj in basis])


w = np.array([0.8, -0.5, 1.2])
print(
    "\nad_{hat(w)} (built from brackets) equals hat(w):",
    np.allclose(ad_matrix(w), hat(w)),
)

# %% [markdown]
# ## 4. The Ad Representation: $\text{Ad}_R = R$
#
# For $R \in SO(3)$, the adjoint is $\text{Ad}_R(\hat\omega) = R\,\hat\omega\,R^\top$. A short
# computation shows $R\,\hat\omega\,R^\top = \widehat{R\omega}$, so **in the basis
# $\{G_1,G_2,G_3\}$ the adjoint matrix is just $R$ itself**:
#
# $$\text{Ad}_R = R$$
#
# Geometrically: conjugating an angular-velocity generator by $R$ rotates its axis by $R$.


# %%
def Ad_matrix(R):
    """Ad_R as a 3x3 matrix in basis {G1,G2,G3}: column j = vee(R Gj Rᵀ)."""
    basis = [G1, G2, G3]
    return np.column_stack([vee(R @ Gj @ R.T) for Gj in basis])


R = exp_so3(np.array([0.4, 0.9, -0.3]))
print("Ad_R (from conjugation R·Gj·Rᵀ) equals R itself:", np.allclose(Ad_matrix(R), R))
# the key identity R·hat(w)·Rᵀ = hat(R w)
w = np.array([1.0, -2.0, 0.5])
print("R·hat(w)·Rᵀ == hat(R·w):", np.allclose(R @ hat(w) @ R.T, hat(R @ w)))

# --- Visualize Ad_R: it rotates an angular-velocity vector's axis ---
fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(
    "Ad_R rotates the axis of a generator:  Ad_R(ω̂) = (Rω)̂", color="#ccc", fontsize=10
)
ax.set_facecolor("#1a1a2e")
w = np.array([0.0, 0.0, 1.2])  # angular velocity about z
R = exp_so3(np.array([0.0, 1.1, 0.0]))  # rotate about y
Adw = R @ w  # the rotated axis
ax.quiver(
    0, 0, 0, *w, color=BLUE, lw=3, arrow_length_ratio=0.12, label="ω (original axis)"
)
ax.quiver(
    0, 0, 0, *Adw, color=ORANGE, lw=3, arrow_length_ratio=0.12, label="Ad_R(ω) = Rω"
)
draw_frame_3d(ax, R, scale=0.9, alpha=0.6, lw=1.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_box_aspect((1, 1, 1))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. The Key Relationship: $\text{Ad}_{e^X} = e^{\text{ad}_X}$
#
# $$\boxed{\text{Ad}_{e^{\hat\omega}} = e^{\text{ad}_{\hat\omega}}}$$
#
# For SO(3) this is a remarkable *tautology in disguise*: the left side is
# $\text{Ad}_{R} = R = e^{\hat\omega}$ (Rodrigues), and the right side is
# $e^{\text{ad}_{\hat\omega}} = e^{\hat\omega}$ (because $\text{ad}_{\hat\omega} = \hat\omega$).
# Both reduce to the *same* Rodrigues exponential. We verify numerically.

# %%
print("Verifying  Ad_{exp(X)} = exp(ad_X)  for X = hat(w):")
print("=" * 54)
for w, desc in [
    (np.array([1.2, 0, 0]), "rotation about x"),
    (np.array([0, 0, 2.0]), "rotation about z"),
    (np.array([0.8, -0.6, 0.4]), "generic axis"),
    (np.array([np.pi, np.pi / 2, 0]), "large rotation"),
]:
    LHS = Ad_matrix(exp_so3(w))  # Ad of the group element exp(hat w)
    RHS = expm(ad_matrix(w))  # exp of ad_{hat w}
    err = np.linalg.norm(LHS - RHS)
    print(
        f"  w = {np.round(w, 3)} ({desc}):  ||Ad_exp − exp_ad||_F = {err:.2e}  →  match: {np.allclose(LHS, RHS)}"
    )

# %% [markdown]
# ## 6. Non-commutativity and the Bracket
#
# The bracket measures the infinitesimal failure of rotations to commute. Rotating about x then
# y is not the same as y then x; to second order, the discrepancy is governed by $[\,\hat x,\hat y\,]=\hat z$.

# %%
fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(
    "Order matters: Rx(θ)·Ry(θ) ≠ Ry(θ)·Rx(θ)\nthe gap between endpoints ∝ [x̂, ŷ] = ẑ",
    color="#ccc",
    fontsize=10,
)
ax.set_facecolor("#1a1a2e")
u, vv = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
ax.plot_surface(
    np.cos(u) * np.sin(vv), np.sin(u) * np.sin(vv), np.cos(vv), color="#333", alpha=0.12
)
p0 = np.array([0.6, 0.2, 0.78])
p0 = p0 / np.linalg.norm(p0)
th = 1.1
# path 1: Rx then Ry
path1 = [exp_so3(t * np.array([1.0, 0, 0])) @ p0 for t in np.linspace(0, th, 30)]
mid1 = exp_so3(th * np.array([1.0, 0, 0]))
path1 += [
    exp_so3(t * np.array([0, 1.0, 0])) @ mid1 @ p0 for t in np.linspace(0, th, 30)
]
path1 = np.array(path1)
# path 2: Ry then Rx
path2 = [exp_so3(t * np.array([0, 1.0, 0])) @ p0 for t in np.linspace(0, th, 30)]
mid2 = exp_so3(th * np.array([0, 1.0, 0]))
path2 += [
    exp_so3(t * np.array([1.0, 0, 0])) @ mid2 @ p0 for t in np.linspace(0, th, 30)
]
path2 = np.array(path2)
ax.plot(*path1.T, color=BLUE, lw=2.5, label="Rx then Ry")
ax.plot(*path2.T, color=ORANGE, lw=2.5, label="Ry then Rx")
ax.scatter(*path1[-1], color=BLUE, s=80, marker="*", zorder=6)
ax.scatter(*path2[-1], color=ORANGE, s=80, marker="*", zorder=6)
ax.plot(*np.array([path1[-1], path2[-1]]).T, color=GREEN, lw=3, label="gap ∝ [x̂, ŷ]")
ax.scatter(*p0, color="white", s=50, zorder=5)
ax.set_box_aspect((1, 1, 1))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | | **ad** | **Ad** |
# |---|---|---|
# | Domain | Lie algebra $\mathfrak{so}(3)$ (skew matrices) | Lie group SO(3) (rotations) |
# | Formula | $\text{ad}_{\hat\omega}(\hat\eta) = [\hat\omega,\hat\eta]$ | $\text{Ad}_R(\hat\omega) = R\hat\omega R^\top$ |
# | In coordinates | $\text{ad}_{\hat\omega} = \hat\omega$ (= cross product) | $\text{Ad}_R = R$ |
# | Bracket = ? | **Yes**, and it equals $a\times b$ | No |
# | Connection | $\text{Ad}_{e^X} = e^{\text{ad}_X}$ | both reduce to Rodrigues $e^{\hat\omega}$ |
#
# The key facts for SO(3):
# - The hat map $\wedge$ turns the **cross product** into the **Lie bracket**:
#   $[\hat a, \hat b] = \widehat{a\times b}$.
# - $\text{ad}_{\hat\omega} = \hat\omega$ and $\text{Ad}_R = R$ — the adjoints *are* the algebra
#   and group elements themselves. This coincidence is special to SO(3).
# - The exponential is **Rodrigues' formula**, a closed form that exists only because
#   $\hat\omega^3 = -\theta^2\hat\omega$; in general one sums the matrix-exponential series.
