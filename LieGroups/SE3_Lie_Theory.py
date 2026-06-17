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
# # Lie Groups and Lie Algebras: SE(3) Illustrated
#
# **SE(3)** is the special Euclidean group in 3D — the group of rigid body motions in space:
# rotations *and* translations. It is the 6-dimensional Lie group at the heart of robotics
# (poses, kinematics) and 3D vision (camera extrinsics). It combines the non-commutativity of
# SO(3) with translations, and its one-parameter subgroups are **screw motions**.
#
# This notebook illustrates:
# 1. The group structure of SE(3) (4×4 homogeneous matrices)
# 2. The Lie algebra $\mathfrak{se}(3)$ — **twists** $\xi = (\omega, v) \in \mathbb{R}^6$
# 3. The Lie bracket (**ad**) as a 6×6 matrix
# 4. The adjoint representation (**Ad**) as a 6×6 matrix
# 5. The relationship $\text{Ad}_{e^X} = e^{\text{ad}_X}$
# 6. Non-commutativity: how the bracket $[X, Y]$ measures the gap between $e^{sX}e^{tY}$ and $e^{tY}e^{sX}$
#
# Throughout, 3D visualizations show the geometry — including screw motions (the exponential of a
# twist) and the order-dependence gap governed by the Lie bracket.
#
# **Convention:** a twist is ordered $\xi = (\omega, v)$ — angular part first, then linear.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

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


def hat3(w):
    """so(3) hat map: 3-vector -> skew-symmetric 3x3 matrix."""
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def vee3(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


# %% [markdown]
# ## 1. SE(3): The Group
#
# A group element $g = (R, t) \in SE(3)$ is a $4\times4$ homogeneous matrix
#
# $$g = \begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix},\qquad R \in SO(3),\; t \in \mathbb{R}^3$$
#
# acting on homogeneous points $(x, 1)^\top$ by rotating then translating. Composition is matrix
# multiplication; the inverse is $(R, t)^{-1} = (R^\top, -R^\top t)$.


# %%
def se3_element(R, t):
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = t
    return g


def se3_inverse(g):
    R, t = g[:3, :3], g[:3, 3]
    return se3_element(R.T, -R.T @ t)


def exp_so3(w):
    """Rodrigues' formula for exp(hat3(w))."""
    theta = np.linalg.norm(w)
    W = hat3(w)
    if theta < 1e-12:
        return np.eye(3) + W
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1 - np.cos(theta)) / theta**2) * (W @ W)
    )


def draw_frame_3d(ax, g, scale=0.6, alpha=1.0, lw=2.5, labels=False):
    """Draw the coordinate frame of an SE(3) pose g at its translation."""
    o = g[:3, 3]
    R = g[:3, :3]
    for i, (col, label) in enumerate(zip([RED, GREEN, BLUE], ["x", "y", "z"])):
        axis = R[:, i] * scale
        ax.quiver(*o, *axis, color=col, lw=lw, alpha=alpha, arrow_length_ratio=0.15)
        if labels:
            ax.text(*(o + axis * 1.1), label, color=col, fontsize=9)


g1 = se3_element(exp_so3([0, 0, np.pi / 4]), np.array([1.0, 0.5, 0.0]))
g2 = se3_element(exp_so3([np.pi / 3, 0, 0]), np.array([0.0, 1.0, 0.5]))
print("g1·g2 ≠ g2·g1  (SE(3) is non-commutative):")
print("g1·g2 translation =", np.round((g1 @ g2)[:3, 3], 3))
print("g2·g1 translation =", np.round((g2 @ g1)[:3, 3], 3))
print("inverse check g1·g1⁻¹ = I:", np.allclose(g1 @ se3_inverse(g1), np.eye(4)))

# --- Visualize a rigid motion of a frame + object ---
fig = plt.figure(figsize=(13, 5.5))
fig.suptitle("SE(3): Rigid Body Motions in 3D", color="white", fontsize=14)
obj = np.array(
    [[0, 0, 0, 1], [0.8, 0, 0, 1], [0, 0.5, 0, 1], [0, 0, 0.4, 1]], dtype=float
).T
for idx, (g, title) in enumerate(
    [(np.eye(4), "Identity e"), (g1 @ g2, "Moved by g₁·g₂")]
):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    p = g @ obj
    ax.scatter(p[0], p[1], p[2], color=ORANGE, s=40)
    draw_frame_3d(ax, g, labels=True)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-0.5, 2)
    ax.set_box_aspect((1, 1, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The Lie Algebra $\mathfrak{se}(3)$: Twists
#
# An element of $\mathfrak{se}(3)$ is a **twist** $\xi = (\omega, v) \in \mathbb{R}^6$, embedded as
# the $4\times4$ matrix
#
# $$\hat\xi = \begin{pmatrix} \hat\omega & v \\ 0 & 0 \end{pmatrix}$$
#
# where $\hat\omega$ is the $\mathfrak{so}(3)$ skew matrix of the angular part and $v$ is the
# linear part. There are 6 generators: 3 rotational, 3 translational.


# %%
def hat6(xi):
    """se(3) hat map: twist (omega, v) -> 4x4 matrix."""
    w, v = xi[:3], xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = hat3(w)
    M[:3, 3] = v
    return M


def vee6(M):
    return np.concatenate([vee3(M[:3, :3]), M[:3, 3]])


# the 6 basis generators
GEN = [hat6(np.eye(6)[i]) for i in range(6)]
print("se(3) has 6 generators: 3 rotational (ω) + 3 translational (v).")
print("\nGenerator G4 (translation along x) =\n", GEN[3].astype(int))
print("\nGenerator G1 (rotation about x) =\n", GEN[0].astype(int))

# %% [markdown]
# ## The exponential map: screw motions and the left Jacobian V
#
# For a twist $\xi = (\omega, v)$ with $\theta = \|\omega\|$, the exponential is
#
# $$e^{\hat\xi} = \begin{pmatrix} R & V v \\ 0 & 1 \end{pmatrix}, \quad
# R = e^{\hat\omega}\ (\text{Rodrigues}), \quad
# V = I + \frac{1-\cos\theta}{\theta^2}\hat\omega + \frac{\theta-\sin\theta}{\theta^3}\hat\omega^2$$
#
# The matrix $V$ (the **left Jacobian** of SO(3)) couples rotation into translation; the result is
# a **screw motion** — simultaneous rotation about and translation along an axis.
#
# > **Caveat — closed forms run out.** SE(3) is still lucky: $R$ and $V$ have closed forms thanks
# > to $\hat\omega^3 = -\theta^2\hat\omega$. But notice the formula is already more elaborate than
# > Rodrigues, and the *inverse* (the log map) and higher groups get worse. For a generic Lie
# > algebra no closed form exists and one simply evaluates the matrix-exponential series with
# > `scipy.linalg.expm` — which we cross-check against below.


# %%
def left_jacobian_so3(w):
    theta = np.linalg.norm(w)
    W = hat3(w)
    if theta < 1e-12:
        return np.eye(3) + 0.5 * W
    return (
        np.eye(3)
        + ((1 - np.cos(theta)) / theta**2) * W
        + ((theta - np.sin(theta)) / theta**3) * (W @ W)
    )


def exp_se3(xi):
    """Closed-form exp of a twist: returns a 4x4 SE(3) matrix (a screw motion)."""
    w, v = xi[:3], xi[3:]
    R = exp_so3(w)
    V = left_jacobian_so3(w)
    return se3_element(R, V @ v)


for xi, desc in [
    (np.array([0, 0, 1.0, 1.0, 0, 0]), "rotate about z + translate x → screw"),
    (np.array([0, 0, 0, 0.5, 1.0, -0.3]), "pure translation"),
    (np.array([0.3, -0.5, 0.8, 0.2, 0.6, -0.4]), "generic twist"),
]:
    closed = exp_se3(xi)
    series = expm(hat6(xi))
    print(f"{desc:38s}  closed form == expm: {np.allclose(closed, series)}")

# --- Visualize a screw motion: exp(t·xi) sweeps a helix ---
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(
    "Screw motion:  g(t) = exp(t·ξ),  ξ = rotation about z + translation along z",
    color="#ccc",
    fontsize=10,
)
ax.set_facecolor("#1a1a2e")
xi = np.array([0, 0, 2.0, 0, 0, 0.25])  # rotate about z, translate along z
ts = np.linspace(0, 3, 60)
p0 = np.array([1.0, 0, 0, 1])
traj = np.array([exp_se3(t * xi) @ p0 for t in ts])
ax.plot(
    traj[:, 0],
    traj[:, 1],
    traj[:, 2],
    color=GREEN,
    lw=2.5,
    label="trajectory of a point",
)
for t in np.linspace(0, 3, 7):
    draw_frame_3d(ax, exp_se3(t * xi), scale=0.4, alpha=0.8, lw=1.5)
ax.scatter(*p0[:3], color="white", s=50, zorder=5)
ax.set_box_aspect((1, 1, 1))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The ad Representation: the 6×6 Bracket Matrix
#
# The Lie bracket of two twists, $[\hat\xi_1, \hat\xi_2] = \hat\xi_1\hat\xi_2 - \hat\xi_2\hat\xi_1$,
# is again a twist. Acting on $\xi_2$, the map $\text{ad}_{\xi_1}$ is the $6\times6$ matrix
#
# $$\text{ad}_{(\omega, v)} = \begin{pmatrix} \hat\omega & 0 \\ \hat v & \hat\omega \end{pmatrix}$$
#
# (lower-triangular block form, in the $(\omega, v)$ ordering).


# %%
def lie_bracket(A, B):
    return A @ B - B @ A


def ad_matrix(xi):
    """ad_xi as a 6x6 matrix in the twist basis: column j = vee6([hat6(xi), Gj])."""
    return np.column_stack([vee6(lie_bracket(hat6(xi), Gj)) for Gj in GEN])


def ad_explicit(xi):
    """Closed-form 6x6 ad matrix: [[ŵ, 0], [v̂, ŵ]]."""
    w, v = xi[:3], xi[3:]
    M = np.zeros((6, 6))
    M[:3, :3] = hat3(w)
    M[3:, :3] = hat3(v)
    M[3:, 3:] = hat3(w)
    return M


xi = np.array([0.5, -0.3, 0.9, 1.0, 0.4, -0.6])
print(
    "ad_xi (from brackets) == closed form [[ŵ,0],[v̂,ŵ]]:",
    np.allclose(ad_matrix(xi), ad_explicit(xi)),
)
print("\nad_xi =\n", np.round(ad_matrix(xi), 3))

# %% [markdown]
# ## 4. The Ad Representation: the 6×6 Adjoint Matrix
#
# For $g = (R, t) \in SE(3)$, the adjoint $\text{Ad}_g(\hat\xi) = g\,\hat\xi\,g^{-1}$ acts on twists
# as the $6\times6$ matrix
#
# $$\text{Ad}_{(R, t)} = \begin{pmatrix} R & 0 \\ \hat t\, R & R \end{pmatrix}$$
#
# The angular part is rotated by $R$; the translation $t$ couples rotation into the linear part
# via $\hat t R$ (the moment of the rotation about the new origin). This is the map that
# transforms a twist from one frame to another — the bread and butter of robot kinematics.


# %%
def Ad_matrix(g):
    """Ad_g as a 6x6 matrix in the twist basis: column j = vee6(g Gj g⁻¹)."""
    g_inv = se3_inverse(g)
    return np.column_stack([vee6(g @ Gj @ g_inv) for Gj in GEN])


def Ad_explicit(g):
    """Closed-form 6x6 Ad matrix: [[R, 0], [t̂R, R]]."""
    R, t = g[:3, :3], g[:3, 3]
    M = np.zeros((6, 6))
    M[:3, :3] = R
    M[3:, :3] = hat3(t) @ R
    M[3:, 3:] = R
    return M


g = exp_se3(np.array([0.4, -0.2, 0.7, 1.0, 0.5, -0.3]))
print(
    "Ad_g (from conjugation) == closed form [[R,0],[t̂R,R]]:",
    np.allclose(Ad_matrix(g), Ad_explicit(g)),
)
print("\nAd_g =\n", np.round(Ad_matrix(g), 3))

# --- Verify Ad really maps a twist between frames: g·exp(ξ)·g⁻¹ = exp(Ad_g ξ) ---
xi = np.array([0.2, 0.5, -0.4, 0.3, -0.1, 0.6])
lhs = g @ exp_se3(xi) @ se3_inverse(g)
rhs = exp_se3(Ad_matrix(g) @ xi)
print("\ng·exp(ξ)·g⁻¹ == exp(Ad_g·ξ):", np.allclose(lhs, rhs))

# %% [markdown]
# ## 5. The Key Relationship: $\text{Ad}_{e^X} = e^{\text{ad}_X}$
#
# $$\boxed{\text{Ad}_{e^{\hat\xi}} = e^{\text{ad}_{\hat\xi}}}$$
#
# The $6\times6$ adjoint of the group element $e^{\hat\xi}$ equals the matrix exponential of the
# $6\times6$ bracket operator $\text{ad}_{\hat\xi}$. We verify numerically across several twists.

# %%
print("Verifying  Ad_{exp(ξ)} = exp(ad_ξ)  (6×6 matrices):")
print("=" * 56)
for xi, desc in [
    (np.array([0, 0, 1.2, 0, 0, 0]), "pure rotation (z)"),
    (np.array([0, 0, 0, 1.0, 0.5, -0.3]), "pure translation"),
    (np.array([0.6, 0, 0, 1.0, 0, 0]), "screw about x"),
    (np.array([0.3, -0.5, 0.8, 0.2, 0.6, -0.4]), "generic twist"),
]:
    LHS = Ad_matrix(exp_se3(xi))  # Ad of the group element exp(hat xi)
    RHS = expm(ad_matrix(xi))  # exp of ad_xi
    err = np.linalg.norm(LHS - RHS)
    print(
        f"  {desc:20s}:  ||Ad_exp − exp_ad||_F = {err:.2e}  →  match: {np.allclose(LHS, RHS)}"
    )

# %% [markdown]
# ## 6. Non-commutativity and the Bracket
#
# As in SE(2) and SO(3), the Lie bracket measures the *infinitesimal failure of two motions to
# commute*. Take two twists $X$ (rotation about $z$) and $Y$ (translation along $x$). Applying
# them in the two orders, $e^{sX}e^{tY}$ vs $e^{tY}e^{sX}$, lands a tracked point in **different
# places** — and the gap is governed by
#
# $$[X, Y] = \big(\omega_X\times\omega_Y,\; \omega_X\times v_Y + v_X\times\omega_Y\big)
#          = (0,\; \hat z \times \hat x) = (0,\; \hat y),$$
#
# a **translation along $y$** (exactly the SE(2) story, now living in 3D). Quantitatively, the
# residual motion $g_{YX}^{-1}\,g_{XY}$ approaches $e^{st\,[X,Y]}$ as $s,t\to 0$: dividing the
# residual twist by $st$ converges to the bracket twist $[X,Y]$.

# %%
X = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # rotation about z
Y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # translation along x

# bracket twist [X, Y] computed via the ad matrix (column form): ad_X · Y
bracket_XY = ad_matrix(X) @ Y
print("X = rotation about z,   Y = translation along x")
print("[X, Y] = (ω, v) =", np.round(bracket_XY, 3), " → a pure +y translation\n")

# residual motion g_YX^{-1} g_XY ≈ exp(st·[X,Y]); residual twist / (st) → [X,Y] as s,t→0
print("Convergence of the residual twist to [X, Y]  (s = t = ε):")
for eps in [0.5, 0.1, 0.01]:
    g_XY = exp_se3(eps * X) @ exp_se3(eps * Y)  # X then Y
    g_YX = exp_se3(eps * Y) @ exp_se3(eps * X)  # Y then X
    residual = se3_inverse(g_YX) @ g_XY
    residual_twist = vee6(np.real(logm(residual))) / eps**2
    print(f"  ε = {eps:5.2f}:  residual_twist/ε² = {np.round(residual_twist, 3)}")

# --- Visualize: a tracked point under both orderings, and the gap ---
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(
    "Order matters: exp(sX)·exp(tY) ≠ exp(tY)·exp(sX)\nthe endpoint gap ∝ [X, Y] = +y translation",
    color="#ccc",
    fontsize=10,
)
ax.set_facecolor("#1a1a2e")
s = t = 1.4
p0 = np.array([1.0, 0.0, 0.0, 1.0])  # tracked point

# Path 1: X then Y
path1 = [exp_se3(tau * X) @ p0 for tau in np.linspace(0, s, 40)]
mid1 = exp_se3(s * X)
path1 += [mid1 @ exp_se3(tau * Y) @ p0 for tau in np.linspace(0, t, 40)]
path1 = np.array(path1)
# Path 2: Y then X
path2 = [exp_se3(tau * Y) @ p0 for tau in np.linspace(0, t, 40)]
mid2 = exp_se3(t * Y)
path2 += [mid2 @ exp_se3(tau * X) @ p0 for tau in np.linspace(0, s, 40)]
path2 = np.array(path2)

ax.plot(
    path1[:, 0],
    path1[:, 1],
    path1[:, 2],
    color=BLUE,
    lw=2.5,
    label="exp(sX) then exp(tY)",
)
ax.plot(
    path2[:, 0],
    path2[:, 1],
    path2[:, 2],
    color=ORANGE,
    lw=2.5,
    label="exp(tY) then exp(sX)",
)
ax.scatter(*path1[-1, :3], color=BLUE, s=90, marker="*", zorder=6)
ax.scatter(*path2[-1, :3], color=ORANGE, s=90, marker="*", zorder=6)
ax.plot(
    *np.array([path1[-1, :3], path2[-1, :3]]).T,
    color=GREEN,
    lw=3,
    label="gap ∝ [X, Y] (along y)",
)
ax.scatter(*p0[:3], color="white", s=50, zorder=5)
ax.text(*p0[:3], "  start", color="white", fontsize=9)
ax.set_box_aspect((1, 1, 1))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | | **ad** | **Ad** |
# |---|---|---|
# | Domain | Lie algebra $\mathfrak{se}(3)$ (twists) | Lie group SE(3) (poses) |
# | Formula | $\text{ad}_\xi(\eta) = [\hat\xi,\hat\eta]$ | $\text{Ad}_g(\hat\xi) = g\hat\xi g^{-1}$ |
# | As a matrix | $\begin{pmatrix}\hat\omega & 0\\ \hat v & \hat\omega\end{pmatrix}$ (6×6) | $\begin{pmatrix}R & 0\\ \hat t R & R\end{pmatrix}$ (6×6) |
# | Role | infinitesimal twist interaction | transform a twist between frames |
# | Connection | $\text{Ad}_{e^X} = e^{\text{ad}_X}$ | both are 6×6 exponentials |
#
# The key ideas for SE(3):
# - Elements of $\mathfrak{se}(3)$ are **twists** $(\omega, v)$; their exponential is a **screw motion**.
# - The exp map needs the **left Jacobian $V$** to couple rotation into translation — a closed
#   form that exists only because SO(3)'s generators satisfy $\hat\omega^3 = -\theta^2\hat\omega$.
#   Not every Lie algebra is so lucky; in general you sum the matrix-exponential series.
# - Both adjoints are $6\times6$ **block-triangular** matrices; $\text{Ad}_g$ is precisely the
#   operator robotics uses to move twists/wrenches between coordinate frames.
