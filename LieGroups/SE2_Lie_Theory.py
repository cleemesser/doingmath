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

# %% [markdown] id="W0pD7aSz-uzZ"
# # Lie Groups and Lie Algebras: SE(2) Illustrated
#
# **SE(2)** is the special Euclidean group in 2D — the group of rigid body motions in the plane: rotations and translations. It is the simplest non-trivial example of a Lie group with a non-commutative structure and non-trivial adjoint action.
#
# This notebook illustrates:
# 1. The group structure of SE(2)
# 2. The Lie algebra 𝔰𝔢(2) and the Lie bracket (**ad**)
# 3. The adjoint representation of the group (**Ad**)
# 4. The relationship: $\text{Ad}_{e^X} = e^{\text{ad}_X}$
# 5. Geometric visualizations of all of the above

# %% id="mNh6591H-uza"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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


# %% [markdown] id="xyrF-mLE-uzb"
# ## 1. SE(2): The Group
#
# A group element $g = (R, v) \in SE(2)$ is a 3×3 matrix:
#
# $$g = \begin{pmatrix} \cos\theta & -\sin\theta & v_x \\ \sin\theta & \cos\theta & v_y \\ 0 & 0 & 1 \end{pmatrix}$$
#
# This acts on homogeneous points $p = (x, y, 1)^T$ by rotating then translating.


# %% colab={"base_uri": "https://localhost:8080/", "height": 895} id="d6iIC3m5-uzc" outputId="98d8371a-7bb7-4b6a-b66b-7402bff822cc"
def rotation_matrix(theta):
    """2x2 rotation matrix for angle theta (radians)"""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def se2_element(theta, vx, vy):
    """SE(2) group element as 3x3 matrix"""
    R = rotation_matrix(theta)
    g = np.eye(3)
    g[:2, :2] = R
    g[:2, 2] = [vx, vy]
    return g


def se2_inverse(g):
    """Inverse of SE(2) element: (R,v)^{-1} = (R^T, -R^T v)"""
    R = g[:2, :2]
    v = g[:2, 2]
    g_inv = np.eye(3)
    g_inv[:2, :2] = R.T
    g_inv[:2, 2] = -R.T @ v
    return g_inv


def draw_frame(ax, g, color="white", label="", alpha=1.0, scale=0.4):
    """Draw a coordinate frame transformed by SE(2) element g"""
    origin = g[:2, 2]
    x_axis = g[:2, :2] @ np.array([scale, 0])
    y_axis = g[:2, :2] @ np.array([0, scale])
    ax.annotate(
        "",
        xy=origin + x_axis,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color=RED, lw=2, alpha=alpha),
    )
    ax.annotate(
        "",
        xy=origin + y_axis,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=2, alpha=alpha),
    )
    ax.plot(*origin, "o", color=color, ms=6, alpha=alpha)
    if label:
        ax.text(
            origin[0] + 0.05,
            origin[1] + 0.05,
            label,
            color=color,
            fontsize=11,
            alpha=alpha,
        )


# --- Visualize group action ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "SE(2): Rigid Body Transformations in the Plane", fontsize=14, color="white", y=1.01
)

# A simple shape (triangle)
triangle = np.array([[0.0, 0.3, 0.15, 0.0], [0.0, 0.0, 0.3, 0.0], [1.0, 1.0, 1.0, 1.0]])

g1 = se2_element(np.pi / 4, 1.0, 0.5)  # rotate 45°, translate
g2 = se2_element(np.pi / 3, -0.5, 1.2)  # rotate 60°, translate
g_composed = g1 @ g2

for ax, (g, label, color, title) in zip(
    axes,
    [
        (g1, "g₁", BLUE, "Single transformation g₁ = (R₄₅°, v₁)"),
        (g_composed, "g₁g₂", PURPLE, "Group composition g₁·g₂"),
    ],
):
    ax.set_title(title, color="#ccc", fontsize=11)
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-0.8, 2.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Original shape
    ax.fill(triangle[0], triangle[1], alpha=0.2, color="white")
    ax.plot(triangle[0], triangle[1], "--", color="white", alpha=0.4, lw=1)
    draw_frame(ax, np.eye(3), color="white", label="e", alpha=0.4)

    # Transformed shape
    t = g @ triangle
    ax.fill(t[0], t[1], alpha=0.4, color=color)
    ax.plot(t[0], t[1], "-", color=color, lw=2)
    draw_frame(ax, g, color=color, label=label)

plt.tight_layout()
plt.show()

print("g1 =\n", np.round(g1, 3))
print("\ng2 =\n", np.round(g2, 3))
print("\ng1 @ g2 =\n", np.round(g_composed, 3))
print("\nNote: g1@g2 ≠ g2@g1 — SE(2) is non-commutative!")
print("g2 @ g1 =\n", np.round(g2 @ g1, 3))

# %% [markdown] id="Pc2be4kR-uzc"
# ## 2. The Lie Algebra 𝔰𝔢(2): Tangent Space at Identity
#
# The Lie algebra consists of 3×3 matrices of the form:
#
# $$X = \begin{pmatrix} 0 & -\omega & u_x \\ \omega & 0 & u_y \\ 0 & 0 & 0 \end{pmatrix}$$
#
# where $\omega$ is the infinitesimal rotation rate and $(u_x, u_y)$ is the infinitesimal translation.
#
# **Basis elements** of 𝔰𝔢(2):

# %% colab={"base_uri": "https://localhost:8080/"} id="Jewk7qCX-uzd" outputId="bca92b70-430c-400c-95c8-3cc3409888ce"
# Basis elements of se(2)
e1 = np.array(
    [
        [0, -1, 0],  # infinitesimal rotation
        [1, 0, 0],
        [0, 0, 0],
    ],
    dtype=float,
)

e2 = np.array(
    [
        [0, 0, 1],  # infinitesimal translation in x
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=float,
)

e3 = np.array(
    [
        [0, 0, 0],  # infinitesimal translation in y
        [0, 0, 1],
        [0, 0, 0],
    ],
    dtype=float,
)


def lie_element(omega, ux, uy):
    """General element of se(2): omega*e1 + ux*e2 + uy*e3"""
    return omega * e1 + ux * e2 + uy * e3


def lie_bracket(X, Y):
    """Lie bracket as matrix commutator [X,Y] = XY - YX"""
    return X @ Y - Y @ X


print("Basis elements of se(2):")
print("\ne1 (infinitesimal rotation):")
print(e1)
print("\ne2 (infinitesimal translation x):")
print(e2)
print("\ne3 (infinitesimal translation y):")
print(e3)

print("\n--- Structure constants via Lie bracket ---")
print("[e1, e2] = XY - YX =")
print(lie_bracket(e1, e2))
print("  → This equals e3 (rotation acting on x-translation gives y-translation)")

print("\n[e1, e3] =")
print(lie_bracket(e1, e3))
print("  → This equals -e2 (rotation acting on y-translation gives -x-translation)")

print("\n[e2, e3] =")
print(lie_bracket(e2, e3))
print("  → Zero: translations commute")

# %% colab={"base_uri": "https://localhost:8080/", "height": 533} id="R_LRR2mN-uzd" outputId="17abb79e-087d-45d8-d632-fc28b47bf6fd"
# Visualize: the Lie algebra as tangent vectors at identity
# Showing how exp(tX) generates curves on SE(2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Lie Algebra se(2): Integral Curves of Basis Elements\n"
    "exp(t·eₖ) traces a path in SE(2); arrows show frame at each t",
    color="white",
    fontsize=12,
)

ts = np.linspace(0, 2 * np.pi, 60)
configs = [
    (e1, "e₁: pure rotation\n(ω=1, u=0)", BLUE),
    (e2, "e₂: pure x-translation\n(ω=0, uₓ=1)", ORANGE),
    (0.5 * e1 + e2, "0.5e₁ + e₂: screw motion\n(rotation + x-translation)", GREEN),
]

for ax, (X, title, color) in zip(axes, configs):
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    origins = []
    for t in ts:
        g = expm(t * X)
        origins.append(g[:2, 2])

    origins = np.array(origins)
    ax.plot(origins[:, 0], origins[:, 1], "-", color=color, lw=2, alpha=0.7)

    # Draw frames at a few points
    for t in np.linspace(0, 2 * np.pi, 8):
        g = expm(t * X)
        draw_frame(ax, g, color=color, alpha=0.8, scale=0.25)

    # Mark start
    ax.plot(0, 0, "o", color="white", ms=8, zorder=5)
    ax.text(0.1, 0.1, "t=0", color="white", fontsize=9)

plt.tight_layout()
plt.show()


# %% [markdown] id="-sp0Ggpt-uzd"
# ## 3. The ad Representation: Lie Bracket as a Linear Map
#
# For $X \in \mathfrak{se}(2)$, the map $\text{ad}_X : \mathfrak{g} \to \mathfrak{g}$ is defined by:
#
# $$\text{ad}_X(Y) = [X, Y]$$
#
# We can represent $\text{ad}_X$ as a **3×3 matrix** (in the basis $\{e_1, e_2, e_3\}$) by computing how $X$ acts on each basis element.


# %% colab={"base_uri": "https://localhost:8080/"} id="2VOwBwK2-uzd" outputId="6333360a-a799-423d-812b-17edec285c8d"
def ad_matrix(X):
    """
    Compute the 3x3 matrix of ad_X in the basis {e1, e2, e3}.
    Each column j is the coordinate vector of [X, e_j] in this basis.
    """
    basis = [e1, e2, e3]
    mat = np.zeros((3, 3))
    for j, ej in enumerate(basis):
        bracket = lie_bracket(X, ej)  # [X, e_j]
        # Extract coordinates: omega is [1,0] entry, ux is [0,2], uy is [1,2]
        omega = bracket[1, 0]
        ux = bracket[0, 2]
        uy = bracket[1, 2]
        mat[:, j] = [omega, ux, uy]
    return mat


# For a general element X = omega*e1 + ux*e2 + uy*e3
# ad_X has a known closed form:
def ad_matrix_explicit(omega, ux, uy):
    """Explicit matrix of ad_X for X = (omega, ux, uy) in se(2) basis"""
    return np.array([[0, 0, 0], [uy, 0, -omega], [-ux, omega, 0]])


# Test with a specific X
omega_test, ux_test, uy_test = 1.0, 0.5, -0.3
X_test = lie_element(omega_test, ux_test, uy_test)

ad_numerical = ad_matrix(X_test)
ad_explicit = ad_matrix_explicit(omega_test, ux_test, uy_test)

print(f"X = {omega_test}·e₁ + {ux_test}·e₂ + {uy_test}·e₃")
print("\nad_X (computed from brackets):")
print(np.round(ad_numerical, 4))
print("\nad_X (explicit formula):")
print(np.round(ad_explicit, 4))
print("\nMatch:", np.allclose(ad_numerical, ad_explicit))

print("\n--- Geometric interpretation ---")
print("ad_X acts on infinitesimal generators Y by [X,Y].")
print("For X = pure rotation (e₁):")
print("  ad_{e₁}(e₂) = [e₁,e₂] = e₃  → rotation maps x-translation to y-translation")
print("  ad_{e₁}(e₃) = [e₁,e₃] = -e₂ → and y-translation to -x-translation")
print("  ad_{e₁}(e₁) = [e₁,e₁] = 0   → rotation commutes with itself")


# %% [markdown] id="w5iGVHwP-uze"
# ## 4. The Ad Representation: Group Acting on Its Algebra
#
# For $g = (R, v) \in SE(2)$, the adjoint representation is:
#
# $$\text{Ad}_g(X) = g X g^{-1}$$
#
# This maps algebra elements to algebra elements. As a 3×3 matrix (in the $\{e_1, e_2, e_3\}$ basis):
#
# $$\text{Ad}_{(R,v)} = \begin{pmatrix} 1 & 0 & 0 \\ v_y & R_{11} & R_{12} \\ -v_x & R_{21} & R_{22} \end{pmatrix}$$
#
# Notice: the rotation part **R acts on the translation part of the algebra**, but leaves the rotation generator invariant.


# %% colab={"base_uri": "https://localhost:8080/"} id="HtKV2OOU-uze" outputId="d504987a-f070-4229-c901-c47e4b182030"
def Ad_matrix(g):
    """
    Compute the 3x3 matrix of Ad_g in the basis {e1, e2, e3}.
    Ad_g(Y) = g Y g^{-1}
    """
    g_inv = se2_inverse(g)
    basis = [e1, e2, e3]
    mat = np.zeros((3, 3))
    for j, ej in enumerate(basis):
        result = g @ ej @ g_inv
        # Extract coordinates
        omega = result[1, 0]
        ux = result[0, 2]
        uy = result[1, 2]
        mat[:, j] = [omega, ux, uy]
    return mat


def Ad_matrix_explicit(theta, vx, vy):
    """Explicit closed-form Ad matrix for SE(2)"""
    R = rotation_matrix(theta)
    return np.array([[1, 0, 0], [vy, R[0, 0], R[0, 1]], [-vx, R[1, 0], R[1, 1]]])


# Test
theta_g, vx_g, vy_g = np.pi / 4, 1.0, 0.5
g = se2_element(theta_g, vx_g, vy_g)

Ad_num = Ad_matrix(g)
Ad_exp = Ad_matrix_explicit(theta_g, vx_g, vy_g)

print(f"g = (θ=π/4, v=({vx_g},{vy_g}))")
print("\nAd_g (from conjugation):")
print(np.round(Ad_num, 4))
print("\nAd_g (explicit formula):")
print(np.round(Ad_exp, 4))
print("\nMatch:", np.allclose(Ad_num, Ad_exp))

print("\n--- What does Ad_g do to each basis element? ---")
g_inv = se2_inverse(g)
for name, ej in [("e₁ (rotation)", e1), ("e₂ (x-transl)", e2), ("e₃ (y-transl)", e3)]:
    result = g @ ej @ g_inv
    omega = result[1, 0]
    ux = result[0, 2]
    uy = result[1, 2]
    print(f"  Ad_g({name}) → ω={omega:.3f}, uₓ={ux:.3f}, u_y={uy:.3f}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 613} id="QiUjkvYO-uze" outputId="9d94764a-e3c6-4b71-bb76-4e1f108b51d2"
# Visualize what Ad_g does geometrically:
# A tangent vector Y at identity, conjugated by g, is a tangent vector at identity
# but 'reinterpreted' from g's perspective

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Ad_g: How a group element g conjugates algebra elements\n"
    "Blue: original generator Y; Orange: Ad_g(Y) = g Y g⁻¹",
    color="white",
    fontsize=12,
)

Y = lie_element(0, 1, 0.3)  # a mixed translation generator

group_elements = [
    (se2_element(0, 1.5, 0), "g = pure translation\n(θ=0, v=(1.5,0))"),
    (se2_element(np.pi / 3, 0, 0), "g = pure rotation\n(θ=60°, v=0)"),
    (
        se2_element(np.pi / 4, 1.0, 0.5),
        "g = rotation + translation\n(θ=45°, v=(1,0.5))",
    ),
]


def extract_coords(X):
    return X[1, 0], X[0, 2], X[1, 2]  # omega, ux, uy


for ax, (g, title) in zip(axes, group_elements):
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.0, 2.0)
    ax.set_aspect("equal")
    ax.grid(True)

    g_inv = se2_inverse(g)
    AdY = g @ Y @ g_inv

    # Original Y as flow arrows at identity
    omega_Y, ux_Y, uy_Y = extract_coords(Y)
    omega_A, ux_A, uy_A = extract_coords(AdY)

    # Show Y and Ad_g(Y) as vectors anchored at origin
    ax.annotate(
        "",
        xy=(ux_Y, uy_Y),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.5),
    )
    ax.text(ux_Y + 0.05, uy_Y + 0.05, "Y", color=BLUE, fontsize=12)

    ax.annotate(
        "",
        xy=(ux_A, uy_A),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5),
    )
    ax.text(ux_A + 0.05, uy_A + 0.05, "Ad_g(Y)", color=ORANGE, fontsize=10)

    # Show g's frame
    draw_frame(ax, g, color=GREEN, label="g", scale=0.3)
    ax.plot(0, 0, "o", color="white", ms=7, zorder=5)
    ax.text(0.05, -0.15, "e", color="white", fontsize=11)

    # Annotation: rotation component
    ax.text(
        -0.4,
        1.7,
        f"ω(Y)={omega_Y:.2f}\nω(Ad_gY)={omega_A:.2f}",
        color="#aaa",
        fontsize=8,
    )

plt.tight_layout()
plt.show()

print("Key observation:")
print("- Pure translation does NOT rotate the translation part of Y")
print("  (but it does mix the rotation into translation via the v×ω term)")
print("- Pure rotation ROTATES the translation part of Y")
print("- The rotation component ω is ALWAYS preserved by Ad_g (top-left entry is 1)")

# %% [markdown] id="CFBwzgLP-uzf"
# ## 5. The Key Relationship: Ad and ad Connected by Exponential Map
#
# $$\boxed{\text{Ad}_{e^X} = e^{\text{ad}_X}}$$
#
# This is the fundamental theorem connecting the group and algebra adjoints. Let's verify it numerically.

# %% colab={"base_uri": "https://localhost:8080/"} id="7RkefFni-uzf" outputId="9b9c59f5-7614-4af9-d460-caf2748e056e"
print("Verifying: Ad_{exp(X)} = exp(ad_X)")
print("=" * 50)

test_cases = [
    (1.2, 0.0, 0.0, "X = 1.2·e₁ (pure rotation generator)"),
    (0.0, 1.5, 0.5, "X = 1.5·e₂ + 0.5·e₃ (pure translation generator)"),
    (0.8, 0.6, -0.4, "X = 0.8·e₁ + 0.6·e₂ - 0.4·e₃ (mixed generator)"),
    (np.pi / 3, 1.0, 1.0, "X = π/3·e₁ + e₂ + e₃ (screw motion)"),
]

for omega, ux, uy, desc in test_cases:
    X = lie_element(omega, ux, uy)

    # Left side: Ad_{exp(X)}
    g_exp = expm(X)  # exp(X) in SE(2) group
    LHS = Ad_matrix(g_exp)  # Ad acting on that group element

    # Right side: exp(ad_X)
    adX = ad_matrix(X)  # ad_X as 3x3 matrix on the algebra
    RHS = expm(adX)  # matrix exponential of ad_X

    match = np.allclose(LHS, RHS, atol=1e-10)
    print(f"\n{desc}")
    print(
        f"  ||Ad_{{exp(X)}} - exp(ad_X)||_F = {np.linalg.norm(LHS - RHS):.2e}  ✓"
        if match
        else f"  MISMATCH: norm = {np.linalg.norm(LHS - RHS):.4f}"
    )

# %% colab={"base_uri": "https://localhost:8080/", "height": 824} id="ZrXNWbEF-uzf" outputId="1890363c-f850-46c5-f6d8-69ed26671850"
# Visualize the relationship along a 1-parameter curve
# As t varies, track how Ad_{exp(tX)} and exp(t·ad_X) evolve

X = lie_element(0.6, 0.8, -0.3)  # a generic element
ts = np.linspace(0, 3, 100)

# Track the (1,1) entry of both matrices as a function of t
LHS_entries = np.zeros((len(ts), 3, 3))
RHS_entries = np.zeros((len(ts), 3, 3))

for i, t in enumerate(ts):
    g_t = expm(t * X)
    LHS_entries[i] = Ad_matrix(g_t)
    RHS_entries[i] = expm(t * ad_matrix(X))

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    r"$Ad_{\exp(tX)}$ vs $\exp(t·ad_X)$: matrix entries as functions of t"
    f"X = 0.6·e₁ + 0.8·e₂ − 0.3·e₃",
    color="white",
    fontsize=12,
)

labels = ["e₁", "e₂", "e₃"]
for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        i, j = row, col
        ax.plot(ts, LHS_entries[:, i, j], color=BLUE, lw=2.5, label=r"$Ad_{\exp(tX)}$")
        ax.plot(
            ts, RHS_entries[:, i, j], "--", color=ORANGE, lw=2, label=r"$\exp(t·ad_X)$"
        )
        ax.set_title(f"Entry ({labels[i]}, {labels[j]})", color="#ccc", fontsize=10)
        ax.grid(True)
        if row == 1 and col == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel("t", fontsize=9)

plt.tight_layout()
plt.show()
print("Blue and orange curves coincide exactly — confirming Ad_{exp(tX)} = exp(t·ad_X)")

# %% [markdown] id="24zoZM_D-uzg"
# ## 6. Putting It All Together: Geometric Summary
#
# A final visualization showing the full chain:
# - A curve $\gamma(t) = e^{tX}$ in SE(2)
# - Its tangent $X \in \mathfrak{se}(2)$ at the identity
# - How **Ad** conjugates tangent vectors, and **ad** brackets them

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} id="XkFYzFNE-uzg" outputId="73e81d62-61e2-423c-f398-2c7aa83d62f8"
fig = plt.figure(figsize=(16, 7))
gs_layout = GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs_layout[0])
ax2 = fig.add_subplot(gs_layout[1])

fig.patch.set_facecolor("#0f0f0f")

# --- LEFT: Two 1-parameter subgroups and their bracket ---
ax1.set_title(
    "Lie bracket as failure of commutativity\nexp(sX)exp(tY) vs exp(tY)exp(sX)",
    color="#ccc",
    fontsize=11,
)
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-1.0, 2.5)
ax1.set_aspect("equal")
ax1.grid(True)

X_gen = lie_element(0.5, 0, 0)  # rotation
Y_gen = lie_element(0, 1, 0)  # x-translation

s, t = 1.0, 1.0
gX = expm(s * X_gen)
gY = expm(t * Y_gen)

# Path 1: exp(sX) then exp(tY)
path1 = []
for tau in np.linspace(0, s, 30):
    path1.append(expm(tau * X_gen)[:2, 2])
p_mid = expm(s * X_gen)
for tau in np.linspace(0, t, 30):
    path1.append((p_mid @ expm(tau * Y_gen))[:2, 2])
path1 = np.array(path1)

# Path 2: exp(tY) then exp(sX)
path2 = []
for tau in np.linspace(0, t, 30):
    path2.append(expm(tau * Y_gen)[:2, 2])
q_mid = expm(t * Y_gen)
for tau in np.linspace(0, s, 30):
    path2.append((q_mid @ expm(tau * X_gen))[:2, 2])
path2 = np.array(path2)

ax1.plot(path1[:, 0], path1[:, 1], "-", color=BLUE, lw=2.5, label="exp(sX)·exp(tY)")
ax1.plot(path2[:, 0], path2[:, 1], "-", color=ORANGE, lw=2.5, label="exp(tY)·exp(sX)")

end1 = (gX @ gY)[:2, 2]
end2 = (gY @ gX)[:2, 2]
ax1.plot(*end1, "*", color=BLUE, ms=14, zorder=6)
ax1.plot(*end2, "*", color=ORANGE, ms=14, zorder=6)

# Show the bracket as the gap
bracket_result = lie_bracket(X_gen, Y_gen)
omega_b, ux_b, uy_b = extract_coords(bracket_result)
ax1.annotate(
    "",
    xy=end2,
    xytext=end1,
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2, connectionstyle="arc3,rad=0.3"),
)
ax1.text(
    (end1[0] + end2[0]) / 2 + 0.1,
    (end1[1] + end2[1]) / 2,
    f"gap ∝ [X,Y]\n= [{omega_b:.2f},{ux_b:.2f},{uy_b:.2f}]",
    color=GREEN,
    fontsize=9,
)

ax1.plot(0, 0, "o", color="white", ms=9, zorder=5)
ax1.text(0.05, -0.2, "identity e", color="white", fontsize=9)
ax1.legend(fontsize=9, loc="upper left")

# --- RIGHT: Summary diagram of the two adjoints ---
ax2.set_facecolor("#0d0d1a")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis("off")
ax2.set_title("Summary: Ad vs ad in SE(2)", color="#ccc", fontsize=12)

box_style = dict(
    boxstyle="round,pad=0.6", facecolor="#1a1a3e", edgecolor="#555", lw=1.5
)

ax2.text(
    2.5,
    6.5,
    "SE(2)",
    fontsize=16,
    color=BLUE,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#001133", edgecolor=BLUE, lw=2),
)
ax2.text(
    7.5,
    6.5,
    "se(2)",
    fontsize=16,
    color=ORANGE,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a0a00", edgecolor=ORANGE, lw=2),
)

ax2.annotate(
    "",
    xy=(6.7, 6.7),
    xytext=(3.3, 6.7),
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
)
ax2.text(5.0, 6.8, r"$\text{log}$", color=GREEN, fontsize=10, ha="center")


ax2.annotate(
    "",
    xy=(6.65, 6.5),
    xytext=(3.2, 6.5),
    arrowprops=dict(arrowstyle="<-", color=PURPLE, lw=2),
)
ax2.text(5.0, 6.3, r"$\exp$", color=PURPLE, fontsize=10, ha="center")

rows = [
    (
        2.5,
        4.8,
        BLUE,
        r"$\text{Ad}_g$ : se(2) → se(2)",
        "Y ↦ g Y g⁻¹",
        "conjugation in group",
    ),
    (7.5, 4.8, ORANGE, r"$\text{ad}_X$ : se(2) → se(2)", "Y ↦ [X, Y]", "Lie bracket"),
    (
        5.0,
        2.8,
        GREEN,
        "Connecting theorem:",
        r"$\text{Ad}_{\exp(X)} = \exp(\text{ad}_X)$",
        "(differentiate Ad at identity)",
    ),
    # (2.5, 1.2, BLUE,   'Ad is a group homomorphism', 'G → GL(𝔤)', ''),
    (
        2.5,
        1.2,
        BLUE,
        "Ad is a group homomorphism",
        r"$G → \text{GL}(\mathfrak{g})$",
        "",
    ),
    # (7.5, 1.2, ORANGE, 'ad is a Lie algebra homomorphism', '𝔤 → 𝔤𝔩(𝔤)', ''),
    (
        7.5,
        1.2,
        ORANGE,
        "ad is a Lie algebra homomorphism",
        r"$\mathfrak{g} → \mathfrak{gl(g)}$",
        "",
    ),
]
for x, y, color, title, sub, note in rows:
    ax2.text(x, y, title, fontsize=10, color=color, ha="center", weight="bold")
    ax2.text(x, y - 0.55, sub, fontsize=9, color="#ccc", ha="center", style="italic")
    if note:
        ax2.text(x, y - 1.0, note, fontsize=8, color="#888", ha="center")

ax2.axhline(y=3.7, color="#333", lw=1, linestyle="--")
# ax2.text(5.0, 3.8, 'SE(2)-specific: rotation ω preserved; translation u rotated by R',
#         color='#aaa', fontsize=8.5, ha='center')
ax2.text(
    5.0,
    3.5,
    "SE(2)-specific: rotation ω preserved; translation u rotated by R",
    color="#aaa",
    fontsize=8.5,
    ha="center",
)

# plt.tight_layout()
plt.show()

# %% [markdown] id="L4J0u9ML-uzg"
# ## Summary
#
# | | **ad** | **Ad** |
# |---|---|---|
# | Domain | Lie algebra 𝔰𝔢(2) | Lie group SE(2) |
# | Formula | $\text{ad}_X(Y) = [X,Y]$ | $\text{Ad}_g(Y) = gYg^{-1}$ |
# | Is the bracket? | **Yes** | No |
# | SE(2) specific | Rotation mixes translation generators | R rotates u-part; ω invariant |
# | Connection | $\text{Ad}_{e^X} = e^{\text{ad}_X}$ | (differentiate Ad at identity → ad) |
#
# The key geometric insight for SE(2):
# - **Rotations and translations don't commute** — rotating then translating lands you somewhere different from translating then rotating.
# - The Lie bracket $[e_1, e_2] = e_3$ captures exactly this: the infinitesimal failure of commutativity between rotation and x-translation is a y-translation.
# - **${\rm Ad}_g$** tells you how a group element "sees" the infinitesimal generators — it rotates the translation part of any algebra element by R, which is precisely the geometric operation of changing the reference frame.

# %% id="Ro8-l2YmBbrT"
