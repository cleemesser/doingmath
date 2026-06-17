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
# # The Lorentz Group: Lie Algebra, Vector Fields, and Physics
#
# The **Lorentz group** is the symmetry group of spacetime in special relativity: the linear maps
# of Minkowski space that preserve the invariant interval. Its restricted (proper, orthochronous)
# component $SO^+(1,3)$ is a 6-dimensional Lie group built from **3 rotations** and **3 boosts**.
#
# This notebook develops it the same way as the rotation groups, and highlights what is new and
# physical:
# 1. Minkowski space, the metric, and what "Lorentz" means.
# 2. **SO⁺(1,1): boosts are hyperbolic rotations** — rapidity, velocity addition, and the
#    *split-complex* numbers (the hyperbolic mirror of SO(2) $\cong$ U(1)).
# 3. The Lie algebra $\mathfrak{so}(1,3)$: 6 generators and the crucial commutator $[K_i, K_j] \sim J_k$.
# 4. The **vector field view**: boost orbits are hyperbolas — the worldlines of constant proper
#    acceleration (Rindler observers).
# 5. **Thomas–Wigner rotation**: two boosts compose to a boost *plus* a rotation.
# 6. The **$SL(2,\mathbb{C})$ double cover** — spinors, the relativistic analogue of $SU(2)\to SO(3)$.
# 7. Why all of this is the backbone of modern physics.
#
# Convention: metric $\eta = \mathrm{diag}(+1, -1, -1, -1)$, coordinates $(t, x, y, z)$, units $c = 1$.

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

ETA = np.diag([1.0, -1.0, -1.0, -1.0])  # Minkowski metric (+,−,−,−)


# %% [markdown]
# ## 1. Minkowski Space and the Lorentz Condition
#
# Spacetime events are 4-vectors $p = (t, x, y, z)$. The **invariant interval**
#
# $$s^2 = \eta_{\mu\nu}\,p^\mu p^\nu = t^2 - x^2 - y^2 - z^2$$
#
# replaces Euclidean length. A **Lorentz transformation** $\Lambda$ is a linear map preserving it,
# i.e. $\Lambda^\top \eta\, \Lambda = \eta$. This is the exact analogue of an orthogonal matrix
# ($R^\top R = I$), but with $\eta$ in place of $I$ — the single sign flip that distinguishes
# *space* from *time* and produces the entire structure of relativity. Its Lie algebra
# $\mathfrak{so}(1,3)$ consists of the $X$ with $X^\top \eta + \eta X = 0$ (**$\eta$-antisymmetric**).


# %%
def is_lorentz(L):
    """Does L preserve the Minkowski metric:  LᵀηL = η ?"""
    return np.allclose(L.T @ ETA @ L, ETA)


def in_algebra(X):
    """Is X in so(1,3):  Xᵀη + ηX = 0 ?"""
    return np.allclose(X.T @ ETA + ETA @ X, 0)


def interval(p):
    return p @ ETA @ p


# light-like, time-like, space-like examples
for p, name in [
    (np.array([1.0, 1.0, 0, 0]), "light-like (on the cone)"),
    (np.array([2.0, 0.5, 0, 0]), "time-like (inside)"),
    (np.array([0.5, 2.0, 0, 0]), "space-like (outside)"),
]:
    print(f"s² = {interval(p):+.2f}   {name}")

# %% [markdown]
# ## 2. SO⁺(1,1): Boosts are Hyperbolic Rotations
#
# Strip to one space + one time dimension. A **boost** with rapidity $\varphi$ is
#
# $$\Lambda(\varphi) = \begin{pmatrix} \cosh\varphi & \sinh\varphi \\ \sinh\varphi & \cosh\varphi \end{pmatrix},$$
#
# which preserves $t^2 - x^2$ exactly as a rotation $\left(\begin{smallmatrix}\cos&-\sin\\\sin&\cos\end{smallmatrix}\right)$
# preserves $x^2 + y^2$ — hyperbolic functions replace circular ones. Rapidities **add**:
# $\Lambda(\varphi_1)\Lambda(\varphi_2) = \Lambda(\varphi_1 + \varphi_2)$ (the boost group is abelian,
# 1-dimensional — the SO(2) of relativity). The physical velocity is $v = \tanh\varphi$, so
# *velocities do not add — rapidities do*, giving the relativistic velocity-addition law.

# %%
eta2 = np.diag([1.0, -1.0])


def boost(phi):
    """SO⁺(1,1) boost with rapidity phi."""
    return np.array([[np.cosh(phi), np.sinh(phi)], [np.sinh(phi), np.cosh(phi)]])


# preserves the interval; rapidities add; velocity = tanh(rapidity)
print(
    "Boost preserves t²−x²  (ΛᵀηΛ = η):",
    np.allclose(boost(0.8).T @ eta2 @ boost(0.8), eta2),
)
print(
    "Rapidities add  Λ(a)Λ(b) = Λ(a+b):",
    np.allclose(boost(0.6) @ boost(0.9), boost(1.5)),
)
v1, v2 = np.tanh(0.6), np.tanh(0.9)
print(f"\nVelocity-addition: v₁={v1:.4f}, v₂={v2:.4f}")
print(f"  relativistic (v₁+v₂)/(1+v₁v₂) = {(v1 + v2) / (1 + v1 * v2):.6f}")
print(
    f"  tanh(φ₁+φ₂)                   = {np.tanh(0.6 + 0.9):.6f}   ← same (rapidity adds)"
)

# %% [markdown]
# ### Split-complex numbers: the hyperbolic mirror of $\mathbb{C}$
#
# Recall SO(2) $\cong$ U(1): rotations are multiplication by unit complex numbers $e^{i\theta}$,
# with $i^2 = -1$. Boosts are the **split-complex** (hyperbolic) analogue: numbers $t + jx$ with
# $j^2 = +1$, represented by $\left(\begin{smallmatrix} t & x \\ x & t \end{smallmatrix}\right)$. Then
# $e^{j\varphi} = \cosh\varphi + j\sinh\varphi$ is exactly the boost, and the "modulus"
# $t^2 - x^2 = \det$ is the **interval**. The sign of $j^2$ versus $i^2$ is the algebraic seed of
# the difference between a circle and a hyperbola — between rotation and boost.


# %%
def split(t, x):
    """Split-complex number t + jx as a 2x2 matrix (j² = +1)."""
    return np.array([[t, x], [x, t]])


# multiplication of split-complex numbers = matrix multiplication; modulus = interval
print(
    "Split-complex product law:",
    np.allclose(split(2, 1) @ split(1, 0.5), split(2 * 1 + 1 * 0.5, 2 * 0.5 + 1 * 1)),
)
print("|t+jx|² = t²−x² = det:", np.isclose(np.linalg.det(split(2, 1)), 2**2 - 1**2))
print(
    "e^{jφ} == boost(φ):",
    np.allclose(expm(0.7 * np.array([[0.0, 1.0], [1.0, 0.0]])), boost(0.7)),
)

# the boost generator K (so(1,1)): exp(φK) = coshφ·I + sinhφ·K, with K² = I
K2 = np.array([[0.0, 1.0], [1.0, 0.0]])
print(
    "\nso(1,1) generator K: K² = I (vs G² = −I for rotation):",
    np.allclose(K2 @ K2, np.eye(2)),
)
print(
    "exp(φK) = coshφ·I + sinhφ·K:",
    np.allclose(expm(0.7 * K2), np.cosh(0.7) * np.eye(2) + np.sinh(0.7) * K2),
)

# --- Spacetime diagram: light cone, invariant hyperbolas, a boosted frame ---
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    "SO⁺(1,1): boosts move events along invariant hyperbolas (t vertical, x horizontal)",
    color="white",
    fontsize=12,
)

ax = axes[0]
ax.set_title(
    "Minkowski diagram: light cone + invariant hyperbolas", color="#ccc", fontsize=10
)
xs = np.linspace(-3, 3, 400)
ax.plot(xs, xs, "--", color=ORANGE, lw=1.5)
ax.plot(xs, -xs, "--", color=ORANGE, lw=1.5, label="light cone  t = ±x")
for c in [0.5, 1.0, 1.5, 2.0]:  # time-like hyperbolas t² − x² = c²
    tt = np.sqrt(c**2 + xs**2)
    ax.plot(xs, tt, "-", color=BLUE, lw=1)
    ax.plot(xs, -tt, "-", color=BLUE, lw=1)
for c in [0.5, 1.0, 1.5, 2.0]:  # space-like hyperbolas x² − t² = c²
    xx = np.sqrt(c**2 + np.linspace(-3, 3, 400) ** 2)
    tt = np.linspace(-3, 3, 400)
    ax.plot(xx, tt, "-", color=GREEN, lw=1)
    ax.plot(-xx, tt, "-", color=GREEN, lw=1)
# a single event boosted to several rapidities, tracing a hyperbola
event = np.array([1.5, 0.0])
orbit = np.array([boost(phi) @ event for phi in np.linspace(-1.2, 1.2, 50)])
ax.plot(
    orbit[:, 1], orbit[:, 0], "o-", color=RED, ms=3, lw=1.5, label="boosts of one event"
)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
ax.set_title(
    "A boost tilts the t and x axes toward the light cone", color="#ccc", fontsize=10
)
ax.plot(xs, xs, "--", color=ORANGE, lw=1.2)
ax.plot(xs, -xs, "--", color=ORANGE, lw=1.2)
phi = 0.6
B = boost(phi)
t_axis = B @ np.array([1.0, 0.0])  # image of the time axis  → t' axis
x_axis = B @ np.array([0.0, 1.0])  # image of the space axis → x' axis
for vec, col, lab in [
    (np.array([1.0, 0]), "#888", "t (rest)"),
    (np.array([0, 1.0]), "#888", "x (rest)"),
    (t_axis, BLUE, "t′ (moving)"),
    (x_axis, GREEN, "x′ (moving)"),
]:
    ax.annotate(
        "",
        xy=(vec[1] * 2.5, vec[0] * 2.5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=col, lw=2.2),
    )
    ax.text(vec[1] * 2.7, vec[0] * 2.7, lab, color=col, fontsize=9, ha="center")
ax.text(
    1.5,
    -2.6,
    f"v = tanh φ = {np.tanh(phi):.2f}  (φ = {phi})",
    color="#ccc",
    fontsize=9,
    ha="center",
)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("t")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The Lie Algebra $\mathfrak{so}(1,3)$: Rotations, Boosts, and Their Brackets
#
# In full 3+1 dimensions $\mathfrak{so}(1,3)$ is **6-dimensional**: three rotation generators
# $J_1, J_2, J_3$ (the $\mathfrak{so}(3)$ of space) and three boost generators $K_1, K_2, K_3$.
# Their commutators encode the whole physics:
#
# $$[J_i, J_j] = \varepsilon_{ijk} J_k, \qquad [J_i, K_j] = \varepsilon_{ijk} K_k, \qquad [K_i, K_j] = -\varepsilon_{ijk} J_k.$$
#
# Read them off: **rotations close** (the $\mathfrak{so}(3)$ subalgebra); **boosts rotate as a
# 3-vector** under rotations; and — the punchline — **the commutator of two boosts is a rotation,
# not a boost.** Boosts alone do *not* form a subgroup. That last relation is the algebraic source
# of Thomas–Wigner rotation (§5).


# %%
def gen_rot(i, j):
    """Rotation generator in the (i,j) spatial plane (antisymmetric block)."""
    M = np.zeros((4, 4))
    M[i, j], M[j, i] = -1.0, 1.0
    return M


def gen_boost(i):
    """Boost generator mixing time (axis 0) with spatial axis i (symmetric block)."""
    M = np.zeros((4, 4))
    M[0, i], M[i, 0] = 1.0, 1.0
    return M


J = [gen_rot(2, 3), gen_rot(3, 1), gen_rot(1, 2)]  # J1, J2, J3 (about x, y, z)
K = [gen_boost(1), gen_boost(2), gen_boost(3)]  # K1, K2, K3 (along x, y, z)


def comm(A, B):
    return A @ B - B @ A


# all six generators are genuinely in so(1,3)
print(
    "All 6 generators η-antisymmetric (in so(1,3)):", all(in_algebra(M) for M in J + K)
)
# and exponentiate to Lorentz transformations
print(
    "exp of each generator is a Lorentz transform:",
    all(is_lorentz(expm(0.7 * M)) for M in J + K),
)

# verify the commutation relations
eps = np.zeros((3, 3, 3))
for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    eps[i, j, k] = 1
    eps[j, i, k] = -1
JJ = all(
    np.allclose(comm(J[i], J[j]), sum(eps[i, j, k] * J[k] for k in range(3)))
    for i in range(3)
    for j in range(3)
)
JK = all(
    np.allclose(comm(J[i], K[j]), sum(eps[i, j, k] * K[k] for k in range(3)))
    for i in range(3)
    for j in range(3)
)
KK = all(
    np.allclose(comm(K[i], K[j]), -sum(eps[i, j, k] * J[k] for k in range(3)))
    for i in range(3)
    for j in range(3)
)
print("\n[Jᵢ,Jⱼ] =  εJ  (rotations close):       ", JJ)
print("[Jᵢ,Kⱼ] =  εK  (boosts rotate as vector):", JK)
print("[Kᵢ,Kⱼ] = −εJ  (boost·boost = ROTATION):  ", KK)

# %% [markdown]
# ## 4. The Vector Field View: Boost Orbits are Hyperbolas (Rindler)
#
# As in the vector-field notebook, a generator $X$ defines a field $\xi_X(p) = X p$ on spacetime
# whose flow is $e^{\varphi X} p$. For the boost generator $K_1$ the field on the $(t,x)$ plane is
# $\xi(t,x) = (x, t)$, and its integral curves are the **hyperbolas** $x^2 - t^2 = \text{const}$.
# Physically these are the worldlines of observers undergoing **constant proper acceleration** — the
# *Rindler observers*. A boost is to spacetime what a rotation is to the plane: motion along the
# orbits of the invariant quadratic form.


# %%
def boost_field(p):
    """Fundamental field of K1 on the (t,x) plane: ξ(t,x) = (x, t)."""
    t, x = p
    return np.array([x, t])


# the field is the velocity of the boost flow, and the flow preserves the interval
p0 = np.array([0.5, 1.4])
vel_analytic = boost_field(p0)
vel_numeric = (boost(1e-5) @ p0 - p0) / 1e-5
print("ξ(p) == d/dφ[boost(φ)p]|₀ :", np.allclose(vel_analytic, vel_numeric, atol=1e-4))
print(
    "boost flow preserves t²−x²:",
    np.isclose(
        p0[0] ** 2 - p0[1] ** 2, (boost(1.3) @ p0)[0] ** 2 - (boost(1.3) @ p0)[1] ** 2
    ),
)

# --- Visualize the boost vector field with its hyperbolic integral curves ---
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.set_title(
    "Boost generator K₁ as a vector field on spacetime\nintegral curves = hyperbolas = constant-proper-acceleration worldlines (Rindler)",
    color="#ccc",
    fontsize=9.5,
)
gt, gx = np.meshgrid(np.linspace(-2.5, 2.5, 19), np.linspace(-2.5, 2.5, 19))
U = gx  # ξ_t = x
Vv = gt  # ξ_x = t
ax.quiver(gx, gt, Vv, U, color="#555", alpha=0.8)  # plot x horizontal, t vertical
xs = np.linspace(-2.5, 2.5, 400)
ax.plot(xs, xs, "--", color=ORANGE, lw=1.5)
ax.plot(xs, -xs, "--", color=ORANGE, lw=1.5, label="light cone")
for x0 in [0.6, 1.2, 1.8]:  # Rindler hyperbolas through (t=0, x=±x0)
    orbit = np.array(
        [boost(phi) @ np.array([0.0, x0]) for phi in np.linspace(-1.6, 1.6, 80)]
    )
    ax.plot(orbit[:, 1], orbit[:, 0], "-", color=BLUE, lw=2)
    ax.plot(-orbit[:, 1], orbit[:, 0], "-", color=GREEN, lw=2)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Thomas–Wigner Rotation: the Bracket Made Physical
#
# Because $[K_i, K_j] = -\varepsilon_{ijk} J_k$, performing two boosts in different directions and
# undoing them does **not** return you to the start — the leftover is a pure **rotation**. Using
# the group-commutator picture from the other notebooks,
#
# $$\Lambda(K_1, \varepsilon)\,\Lambda(K_2, \varepsilon)\,\Lambda(K_1, -\varepsilon)\,\Lambda(K_2, -\varepsilon)
#   \;\approx\; e^{\varepsilon^2 [K_1, K_2]} = e^{-\varepsilon^2 J_3},$$
#
# a rotation about $z$ by angle $\sim \varepsilon^2$. This is the **Thomas–Wigner rotation**; its
# atomic-physics consequence (Thomas precession) supplies the famous factor of $\tfrac12$ in
# spin–orbit coupling. We confirm the leftover is a rotation and measure its angle.


# %%
def rot_angle(L):
    """Rotation angle of the spatial 3x3 block of a Lorentz transform that fixes the time axis."""
    R = L[1:, 1:]
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))


print("Group commutator of two boosts → a pure rotation:")
print(f"{'ε':>6} | {'is rotation? (fixes t-axis, ∈SO)':35} | angle/ε²")
for ep in [0.4, 0.2, 0.1, 0.05]:
    C = expm(ep * K[0]) @ expm(ep * K[1]) @ expm(-ep * K[0]) @ expm(-ep * K[1])
    fixes_time = np.allclose(C @ np.array([1.0, 0, 0, 0]), [1, 0, 0, 0], atol=1e-2)
    is_rot = is_lorentz(C) and fixes_time
    print(f"{ep:>6} | {str(is_rot):35} | {rot_angle(C) / ep**2:.4f}")
print(
    "\n→ leftover angle/ε² → 1, matching |[K₁,K₂]| = |J₃|.  Boosts don't commute: they leave a rotation."
)

# a finite, non-infinitesimal example: two perpendicular rapidity-1 boosts
C = expm(1.0 * K[0]) @ expm(1.0 * K[1]) @ expm(-1.0 * K[0]) @ expm(-1.0 * K[1])
print(
    f"Finite case (ε=1): leftover is a rotation of {np.degrees(rot_angle(C)):.1f}° about z."
)

# %% [markdown]
# ## 6. The $SL(2,\mathbb{C})$ Double Cover: Spinors
#
# Just as $SU(2)$ double-covers $SO(3)$, the group $SL(2,\mathbb{C})$ (complex $2\times2$ matrices
# with $\det = 1$) **double-covers** the restricted Lorentz group $SO^+(1,3)$. The dictionary:
# encode a 4-vector as a **Hermitian** matrix
#
# $$P = t\,I + x\,\sigma_1 + y\,\sigma_2 + z\,\sigma_3 = \begin{pmatrix} t+z & x - iy \\ x + iy & t - z \end{pmatrix},
#   \qquad \det P = t^2 - x^2 - y^2 - z^2 = s^2.$$
#
# An $A \in SL(2,\mathbb{C})$ acts by $P \mapsto A P A^\dagger$, which preserves $\det P$ (since
# $\det A = 1$) — hence preserves the interval and induces a Lorentz transformation. Both $A$ and
# $-A$ give the same one (2-to-1). Rotations come from $A = e^{-i\theta\,\sigma\cdot\hat n/2}$
# (unitary, as in $SU(2)$); **boosts** come from $A = e^{\varphi\,\sigma\cdot\hat n/2}$ (Hermitian,
# *non*-unitary) — the half-rapidity is the spinor signature. Spinors are the objects this group
# acts on, and they are the foundation of the Dirac equation.

# %%
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [np.eye(2, dtype=complex), s1, s2, s3]


def four_to_herm(p):
    return p[0] * np.eye(2) + p[1] * s1 + p[2] * s2 + p[3] * s3


def herm_to_four(H):
    return np.array([0.5 * np.real(np.trace(s @ H)) for s in PAULI])


def lorentz_from_sl2c(A):
    """The 4x4 Lorentz transform induced by A ∈ SL(2,ℂ) via P ↦ A P A†."""
    return np.column_stack(
        [herm_to_four(A @ four_to_herm(np.eye(4)[k]) @ A.conj().T) for k in range(4)]
    )


# det P is the interval
p = np.array([2.0, 0.7, -0.4, 0.5])
print(
    "det P == interval s²:",
    np.isclose(np.real(np.linalg.det(four_to_herm(p))), interval(p)),
)

# a spinor BOOST (Hermitian, half-rapidity) induces the 4-vector boost along x
phi = 0.9
A_boost = expm(phi * s1 / 2)
L_boost = lorentz_from_sl2c(A_boost)
print(
    "Hermitian A = exp(φσ₁/2) induces boost K₁ by φ:",
    np.allclose(L_boost, expm(phi * K[0])),
)
print("  (and it IS a Lorentz transform):", is_lorentz(L_boost))

# a spinor ROTATION (unitary, half-angle) induces the 4-vector rotation about z
theta = 1.1
A_rot = expm(-1j * theta * s3 / 2)
print(
    "Unitary A = exp(−iθσ₃/2) induces rotation J₃ by θ:",
    np.allclose(lorentz_from_sl2c(A_rot), expm(theta * J[2])),
)

# the 2-to-1 cover: A and −A give the same Lorentz transform
print(
    "A and −A map to the SAME Lorentz transform (2:1 cover):",
    np.allclose(lorentz_from_sl2c(A_boost), lorentz_from_sl2c(-A_boost)),
)

# %% [markdown]
# ### $\mathfrak{so}(1,3)_\mathbb{C} \cong \mathfrak{su}(2) \oplus \mathfrak{su}(2)$
#
# The complex combinations $A_k = \tfrac12(J_k + i K_k)$ and $B_k = \tfrac12(J_k - i K_k)$ split the
# Lorentz algebra into **two commuting copies of $\mathfrak{su}(2)$**. This is why relativistic
# fields are labelled by a *pair* of spins $(j_L, j_R)$: $(0,0)$ scalar, $(\tfrac12,0)$ and
# $(0,\tfrac12)$ the left/right **Weyl spinors**, $(\tfrac12,\tfrac12)$ the 4-vector.

# %%
A = [0.5 * (J[k] + 1j * K[k]) for k in range(3)]
B = [0.5 * (J[k] - 1j * K[k]) for k in range(3)]
AA = all(
    np.allclose(comm(A[i], A[j]), sum(eps[i, j, k] * A[k] for k in range(3)))
    for i in range(3)
    for j in range(3)
)
BB = all(
    np.allclose(comm(B[i], B[j]), sum(eps[i, j, k] * B[k] for k in range(3)))
    for i in range(3)
    for j in range(3)
)
AB = all(np.allclose(comm(A[i], B[j]), 0) for i in range(3) for j in range(3))
print("[Aᵢ,Aⱼ] = εA  (first su(2)):  ", AA)
print("[Bᵢ,Bⱼ] = εB  (second su(2)): ", BB)
print("[Aᵢ,Bⱼ] = 0   (they commute): ", AB)
print("\n→ so(1,3)_ℂ ≅ su(2) ⊕ su(2): relativistic fields carry two spins (j_L, j_R).")

# %% [markdown]
# ## 7. Summary: Why the Lorentz Group is the Backbone of Physics
#
# | | rotations SO(3) | Lorentz SO⁺(1,3) |
# |---|---|---|
# | preserves | length $x^2+y^2+z^2$ | interval $t^2 - x^2 - y^2 - z^2$ |
# | "rotation" in a plane | circular, $\cos/\sin$ | space–time: **boost**, $\cosh/\sinh$ |
# | additive parameter | angle $\theta$ | **rapidity** $\varphi$ ($v = \tanh\varphi$) |
# | number system | complex $i^2=-1$ | split-complex $j^2=+1$ |
# | bracket surprise | — | $[K_i,K_j] = -\varepsilon J_k$ (boosts → rotation) |
# | double cover | $SU(2)$ | $SL(2,\mathbb{C})$ |
#
# **The physics that falls out:**
# - **Special relativity itself** is the principle that the laws of physics are Lorentz-invariant.
#   Time dilation, length contraction, and the relativity of simultaneity are all just §2's boost
#   tilting the time and space axes.
# - **No simple velocity addition:** rapidity is additive, velocity saturates at $c$ ($\tanh \to 1$).
# - **Causality** is the invariant light-cone structure of §1 — time-like, light-like, space-like.
# - **Thomas precession** (§5) — the $[K,K]\sim J$ rotation — gives the factor $\tfrac12$ in atomic
#   spin–orbit coupling, a measured effect.
# - **Spinors and the Dirac equation:** $SL(2,\mathbb{C})$ (§6) is the spin group of spacetime; the
#   $\mathfrak{su}(2)\oplus\mathfrak{su}(2)$ split classifies *every* relativistic field (scalars,
#   Weyl/Dirac spinors, vectors, the graviton). All of relativistic quantum field theory is
#   representation theory of this one group, extended by translations to the **Poincaré group**.
