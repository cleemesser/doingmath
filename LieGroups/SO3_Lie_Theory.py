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
# 6. Non-commutativity: the bracket as the gap between rotating in two orders
# 7. The double cover $SU(2) \to SO(3)$ — unit quaternions and the $4\pi$ spinor periodicity
#
# 3D visualizations run throughout.

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
# ## 7. SU(2) → SO(3): the Double Cover (Unit Quaternions)
#
# Just as SO(2) *is* the unit complex numbers U(1), SO(3) has a complex-number cousin one
# dimension up: **SU(2)**, the group of $2\times2$ unitary matrices with $\det = 1$, equivalently
# the **unit quaternions** $\{q : \|q\| = 1\} = S^3$. But the relationship is subtler than an
# isomorphism — it is a **2-to-1 covering map**
#
# $$\rho : SU(2) \longrightarrow SO(3), \qquad \rho(q) = \rho(-q).$$
#
# Every rotation has **two** unit-quaternion preimages, $q$ and $-q$. A quaternion encodes a
# rotation by angle $\theta$ about axis $n$ using the **half-angle**
# $q = \cos(\tfrac{\theta}{2}) + \sin(\tfrac{\theta}{2})(n_x i + n_y j + n_z k)$, which is exactly
# why a full $2\pi$ turn sends $q \to -q$ (the famous *spinor sign flip*) and you need $4\pi$ to
# return. At the **algebra** level the cover is a local isomorphism: $\mathfrak{su}(2) \cong
# \mathfrak{so}(3)$, with generators $-i\sigma_k/2$ obeying the *same* cross-product bracket.


# %%
def quat_mul(p, q):
    """Hamilton product of quaternions p, q given as (w, x, y, z)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ]
    )


def axis_angle_to_quat(axis, angle):
    """Unit quaternion for rotation by `angle` about `axis` — note the HALF angle."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * n])


def quat_to_rot(q):
    """Map a unit quaternion (w,x,y,z) to its 3x3 rotation matrix ρ(q)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# (a) ρ reproduces Rodrigues: quaternion rotation == matrix exponential
n, theta = np.array([0.3, -0.7, 1.1]), 1.4
n_hat = n / np.linalg.norm(n)
q = axis_angle_to_quat(n, theta)
print(
    "ρ(q) == exp_so3(θ·n̂) (quaternion agrees with Rodrigues):",
    np.allclose(quat_to_rot(q), exp_so3(theta * n_hat)),
)
print(
    "  (angle θ =",
    round(theta, 2),
    ") encoded with half-angle cos(θ/2) =",
    round(q[0], 4),
)

# (b) homomorphism: quaternion multiplication ↔ rotation composition
p = axis_angle_to_quat([1, 0, 0], 0.9)
r = axis_angle_to_quat([0, 1, 1], 1.3)
print(
    "\nρ(p·q) == ρ(p)·ρ(q) (covering map is a homomorphism):",
    np.allclose(quat_to_rot(quat_mul(p, r)), quat_to_rot(p) @ quat_to_rot(r)),
)

# (c) the 2-to-1 cover: q and −q map to the SAME rotation
print(
    "\nρ(q) == ρ(−q) (two preimages per rotation → 2:1 cover):",
    np.allclose(quat_to_rot(q), quat_to_rot(-q)),
)

# %% [markdown]
# ### SU(2) as $2\times2$ matrices, and $\mathfrak{su}(2) \cong \mathfrak{so}(3)$
#
# Writing $q = w + xi + yj + zk$ as $U(q) = w\,I - i(x\sigma_1 + y\sigma_2 + z\sigma_3)$ (with the
# Pauli matrices $\sigma_k$) realizes the unit quaternions as genuine **SU(2)** matrices —
# unitary, $\det = 1$. Quaternion multiplication becomes matrix multiplication, and the algebra
# generators $E_k = -i\sigma_k/2$ satisfy $[E_j, E_k] = \varepsilon_{jkl}E_l$ — identical to the
# $\mathfrak{so}(3)$ structure constants.

# %%
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)


def quat_to_su2(q):
    """Map a unit quaternion to its SU(2) matrix U = wI − i(xσ1 + yσ2 + zσ3)."""
    w, x, y, z = q
    return w * np.eye(2) - 1j * (x * s1 + y * s2 + z * s3)


U = quat_to_su2(q)
print("U is unitary (U U† = I):", np.allclose(U @ U.conj().T, np.eye(2)))
print("det U = 1 (so U ∈ SU(2)):", np.isclose(np.linalg.det(U), 1.0))
print(
    "quaternion product ↔ SU(2) product:",
    np.allclose(quat_to_su2(quat_mul(p, r)), quat_to_su2(p) @ quat_to_su2(r)),
)

# su(2) generators E_k = −iσ_k/2 share so(3)'s structure constants
E1, E2, E3 = -1j * s1 / 2, -1j * s2 / 2, -1j * s3 / 2
print(
    "\nsu(2) ≅ so(3):  [E1,E2]=E3:",
    np.allclose(E1 @ E2 - E2 @ E1, E3),
    " [E2,E3]=E1:",
    np.allclose(E2 @ E3 - E3 @ E2, E1),
    " [E3,E1]=E2:",
    np.allclose(E3 @ E1 - E1 @ E3, E2),
)

# %% [markdown]
# ### Visualizing the double cover: the $4\pi$ spinor periodicity
#
# Rotate continuously about a fixed axis by an angle $\theta$ growing from $0$ to $4\pi$. The
# rotation $\rho(q)$ returns to the identity every $2\pi$, but the quaternion's scalar part
# $\cos(\theta/2)$ has period $4\pi$: at $\theta = 2\pi$ the rotation is the identity yet the
# quaternion is $-1$. You must turn through $720°$ to bring the quaternion home — the geometric
# heart of why electrons are spin-$\tfrac12$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "SU(2) double-covers SO(3): 360° flips the quaternion's sign, 720° restores it",
    color="white",
    fontsize=12,
)

thetas = np.linspace(0, 4 * np.pi, 400)
qw = np.cos(thetas / 2)  # quaternion scalar part along the path q(θ) about z
# "how far the rotation is from identity": ρ returns every 2π
rot_dist = np.array(
    [
        np.linalg.norm(quat_to_rot(axis_angle_to_quat([0, 0, 1], th)) - np.eye(3))
        for th in thetas
    ]
)

ax = axes[0]
ax.set_title(
    "Quaternion scalar cos(θ/2): period 4π (the spinor)", color="#ccc", fontsize=10
)
ax.plot(thetas, qw, color=BLUE, lw=2.5)
ax.axhline(0, color="#555", lw=0.8)
for th, lab, col in [
    (0, "q=+1", GREEN),
    (2 * np.pi, "q=−1\n(R=I!)", RED),
    (4 * np.pi, "q=+1", GREEN),
]:
    ax.plot(th, np.cos(th / 2), "o", color=col, ms=10, zorder=5)
    ax.annotate(
        lab,
        xy=(th, np.cos(th / 2)),
        xytext=(th, np.cos(th / 2) + 0.25),
        color=col,
        fontsize=9,
        ha="center",
    )
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_xticklabels(["0", "π", "2π\n(360°)", "3π", "4π\n(720°)"])
ax.set_xlabel("rotation angle θ")
ax.grid(True)

ax = axes[1]
ax.set_title(
    "The rotation ρ(q) itself: period 2π (returns twice as fast)",
    color="#ccc",
    fontsize=10,
)
ax.plot(thetas, rot_dist, color=ORANGE, lw=2.5)
for th in [0, 2 * np.pi, 4 * np.pi]:
    ax.axvline(th, color="#555", ls="--", lw=1)
ax.annotate(
    "R = I at 0, 2π, 4π →\nbut q differs by sign at 2π",
    xy=(2 * np.pi, 0.1),
    xytext=(0.6 * np.pi, 1.5),
    color="#ccc",
    fontsize=9,
)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_xticklabels(["0", "π", "2π", "3π", "4π"])
ax.set_xlabel("rotation angle θ")
ax.set_ylabel("‖ρ(q) − I‖")
ax.grid(True)
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
