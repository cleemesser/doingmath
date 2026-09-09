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
# # Divergence and Curl as Pieces of $DF$
#
# ### What the derivative of a vector field is, and how $\operatorname{div}$ and $\operatorname{curl}$ read it
#
# A vector field $F:\mathbb{R}^n \to \mathbb{R}^n$ is special among smooth maps: its **derivative at
# a point is an endomorphism of the tangent space**,
#
# $$DF(p) : T_p\mathbb{R}^n \longrightarrow T_p\mathbb{R}^n .$$
#
# For a general map $f : M \to N$ the derivative goes $T_pM \to T_{f(p)}N$ — two *different* spaces —
# and so it has no trace, no eigenvalues, no symmetric part. Only when domain and codomain are
# identified does $DF(p)$ become a matrix you may legitimately take the trace of. That single fact
# is the source of everything below.
#
# An endomorphism of an inner-product space splits, canonically and irreducibly, into three pieces:
#
# $$DF \;=\; \underbrace{\tfrac{1}{n}(\operatorname{tr} DF)\,I}_{\text{isotropic dilation}}
#            \;+\; \underbrace{S_0}_{\substack{\text{trace-free symmetric}\\ \text{(pure shear / strain)}}}
#            \;+\; \underbrace{A}_{\substack{\text{antisymmetric}\\ \text{(infinitesimal rotation)}}}$$
#
# and the claim of this notebook is that
#
# > **divergence is the first piece and curl is the third**, and *the middle piece is invisible to
# > both*.
#
# In $\mathbb{R}^3$ that middle piece has 5 of the 9 components. So "div and curl" is not a complete
# description of $DF$ — it is a projection onto two of the three irreducible summands of
# $\mathfrak{gl}(n)$ under the orthogonal group. Knowing this tells you exactly what div and curl can
# and cannot see, why curl is a *vector* only in three dimensions, and why div needs less structure
# than curl.
#
# We then take the view of Spivak's *Calculus on Manifolds*: the honest objects are not $\operatorname{div}$
# and $\operatorname{curl}$ but the **exterior derivative $d$** applied to two different forms built
# from $F$. Everything else — the vector-valued curl, the classical Green/Stokes/Gauss theorems — is
# $d$ plus a metric plus an orientation.
#
# Everything asserted here is checked in code: numerically with `np.allclose`, and — where the claim
# is an identity for *all* fields — symbolically with SymPy over generic functions, which is a proof
# rather than a spot check.
#
# **Contents**
#
# 1. The decomposition of $DF$ and the two projections
# 2. What is invariant: why $\operatorname{tr}$ is cheap and the antisymmetric part is expensive
# 3. Divergence = rate of change of volume (Liouville)
# 4. Curl = twice the angular velocity (Cauchy–Stokes)
# 5. The Spivak view: it is all $d$
# 6. Why curl is a vector only when $n = 3$
# 7. What div and curl do *not* see — and what they nevertheless determine
# 8. A coda: Jacobians of learned flows

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import expm, polar

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
# ## Numerical toolkit
#
# A finite-difference Jacobian, the three projections, and the $\mathfrak{so}(3)\cong\mathbb{R}^3$ hat
# map (same convention as the `LieGroups/SO3_Lie_Theory` notebook: $\widehat{w}\,v = w \times v$).


# %%
def jacobian(F, p, h=1e-6):
    """Finite-difference Jacobian, J[i, j] = ∂F_i/∂x_j, of a vector field F at p."""
    p = np.asarray(p, dtype=float)
    n = p.size
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (np.asarray(F(p + e), float) - np.asarray(F(p - e), float)) / (2 * h)
    return J


def sym(J):
    """Symmetric part ½(J + Jᵀ) — the rate-of-strain tensor."""
    return 0.5 * (J + J.T)


def skew(J):
    """Antisymmetric part ½(J − Jᵀ) — the spin tensor."""
    return 0.5 * (J - J.T)


def iso(J):
    """Isotropic part (tr J / n) I — pure dilation."""
    n = J.shape[0]
    return (np.trace(J) / n) * np.eye(n)


def dev(J):
    """Trace-free symmetric part (the deviatoric strain) — what div and curl never see."""
    return sym(J) - iso(J)


def hat(w):
    """ℝ³ → so(3):  hat(w) v = w × v."""
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def vee(A):
    """so(3) → ℝ³, the inverse of hat."""
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def div(F, p, h=1e-6):
    return np.trace(jacobian(F, p, h))


def curl3(F, p, h=1e-6):
    """curl of a field on ℝ³, read off the antisymmetric part: curl F = 2·vee(skew DF)."""
    return 2.0 * vee(skew(jacobian(F, p, h)))


def curl2(F, p, h=1e-6):
    """The scalar curl in the plane: ∂ₓF_y − ∂_yF_x."""
    J = jacobian(F, p, h)
    return J[1, 0] - J[0, 1]


# %% [markdown]
# ## 1. The decomposition of $DF$, and the two projections
#
# Write $J = DF(p)$, an $n\times n$ matrix. The splitting
#
# $$J \;=\; \underbrace{\tfrac{\operatorname{tr}J}{n}I}_{\text{dilation}}
#   \;+\;\underbrace{\left(\tfrac{J+J^{\mathsf T}}{2}-\tfrac{\operatorname{tr}J}{n}I\right)}_{S_0,\ \text{shear}}
#   \;+\;\underbrace{\tfrac{J-J^{\mathsf T}}{2}}_{A,\ \text{rotation}}$$
#
# is the decomposition of $\mathfrak{gl}(n)$ into irreducible representations of $SO(n)$ acting by
# conjugation:
#
# $$\mathfrak{gl}(n)\;=\;\mathbb{R}\,I\;\oplus\;\operatorname{Sym}_0(n)\;\oplus\;\mathfrak{so}(n),
# \qquad n^2 \;=\; 1 \;+\; \left(\tfrac{n(n+1)}{2}-1\right)\;+\;\tfrac{n(n-1)}{2}.$$
#
# In $n=3$ that reads $9 = 1 + 5 + 3$ — a scalar, a quadrupole, and a vector. Physicists meet exactly
# this when a rank-2 Cartesian tensor is reduced to spin $0 \oplus 2 \oplus 1$.
#
# The two classical operators are the two "easy" summands:
#
# $$\boxed{\operatorname{div}F \;=\; \operatorname{tr}DF} \qquad\qquad
#   \boxed{A\,v \;=\; \tfrac12\,(\operatorname{curl}F)\times v \quad\text{in } \mathbb{R}^3}$$
#
# i.e. $\operatorname{curl}F = 2\,\mathrm{vee}(A)$. Divergence *is* the trace; curl *is* (twice) the
# antisymmetric part, repackaged as a vector by the hat map. Nothing else about $DF$ appears.


# %%
# A field with all three parts nonzero: dilation + shear + swirl, plus nonlinearity.
def F3(p):
    x, y, z = p
    return np.array(
        [
            x + 2.0 * y + 0.3 * y * z,
            -1.0 * x + 0.5 * y * y + z**2,
            0.7 * np.sin(2.0 * x) * z - 0.4 * x * y,
        ]
    )


p = np.array([0.4, -0.7, 1.1])
J = jacobian(F3, p)
D, S0, A = iso(J), dev(J), skew(J)

print("J = DF(p) =\n", np.round(J, 6))
print("\n  dilation  (tr J / 3) I =\n", np.round(D, 6))
print("\n  shear     S₀ =\n", np.round(S0, 6))
print("\n  rotation  A =\n", np.round(A, 6))

print("\nJ == D + S₀ + A :", np.allclose(J, D + S0 + A))
print("S₀ symmetric, trace-free :", np.allclose(S0, S0.T), np.isclose(np.trace(S0), 0))
print("A antisymmetric          :", np.allclose(A, -A.T))

# the two projections are exactly div and curl
print("\ndiv F  = tr DF          :", np.isclose(div(F3, p), np.trace(J)))
print("curl F = 2·vee(skew DF) :", np.round(curl3(F3, p), 6))

# and A acts as a cross product with ½ curl
w = 0.5 * curl3(F3, p)
v = np.array([0.3, 1.2, -0.8])
print("A v == ½ curl F × v     :", np.allclose(A @ v, np.cross(w, v)))
print("A == hat(½ curl F)      :", np.allclose(A, hat(w)))

# the three summands are mutually orthogonal in the Frobenius inner product ⟨X,Y⟩ = tr(XᵀY)
fro = lambda X, Y: float(np.trace(X.T @ Y))
print(
    "\nFrobenius-orthogonal: ⟨D,S₀⟩ = %.2e, ⟨D,A⟩ = %.2e, ⟨S₀,A⟩ = %.2e"
    % (fro(D, S0), fro(D, A), fro(S0, A))
)
print(
    "Pythagoras ‖J‖² = ‖D‖² + ‖S₀‖² + ‖A‖² :",
    np.isclose(fro(J, J), fro(D, D) + fro(S0, S0) + fro(A, A)),
)

# %% [markdown]
# The last two checks matter: the splitting is an **orthogonal** decomposition, so
# $\|DF\|^2 = \|D\|^2 + \|S_0\|^2 + \|A\|^2$ is a genuine budget. We can therefore ask what *fraction*
# of the derivative div and curl actually capture at a point.

# %%
budget = {
    "dilation  (div)": fro(D, D),
    "shear     (invisible)": fro(S0, S0),
    "rotation  (curl)": fro(A, A),
}
total = sum(budget.values())
print("How much of ‖DF‖² each summand holds, at p =", p)
for k, v_ in budget.items():
    print(f"  {k:24s} {v_ / total:6.1%}")
print(
    f"\n  seen by div+curl: {(budget['dilation  (div)'] + budget['rotation  (curl)']) / total:.1%}"
)

# %% [markdown]
# ### The three atoms
#
# Every $DF$ is a superposition of three elementary planar motions. Below, each is shown as a field
# together with the image of a small circle of material points carried by its flow $e^{tJ}$ — the
# circle dilates, distorts at fixed area, or turns rigidly.

# %%
atoms = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), "dilation   J = I", BLUE),
    (np.array([[1.0, 0.0], [0.0, -1.0]]), "shear      J = diag(1,−1)", ORANGE),
    (np.array([[0.0, 1.0], [1.0, 0.0]]), "shear      J = [[0,1],[1,0]]", YELLOW),
    (np.array([[0.0, -1.0], [1.0, 0.0]]), "rotation   J = [[0,−1],[1,0]]", GREEN),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5.4))
fig.suptitle(
    "The three atoms of DF:  dilation (div ≠ 0),  two shears (div = curl = 0),  rotation (curl ≠ 0)"
    "\nblue circle = material points at t = 0;  colored curve = the same points at t = 0.4",
    color="white",
    fontsize=11,
)
th = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(th), np.sin(th)])
gx, gy = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
for ax, (M, title, color) in zip(axes, atoms):
    U = M[0, 0] * gx + M[0, 1] * gy
    V = M[1, 0] * gx + M[1, 1] * gy
    ax.quiver(gx, gy, U, V, color="#555", alpha=0.9)
    ax.plot(circle[0], circle[1], color="#bbb", lw=1.4, ls="--")
    img = expm(0.4 * M) @ circle
    ax.plot(img[0], img[1], "-", color=color, lw=2.6)
    # material vectors, to show what happens to a frame
    for e, c in [(np.array([1.0, 0.0]), RED), (np.array([0.0, 1.0]), PURPLE)]:
        w_ = expm(0.4 * M) @ e
        ax.annotate(
            "", xy=w_, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2)
        )
    ax.set_title(
        f"{title}\ndiv = {np.trace(M):+.0f}   curl = {M[1, 0] - M[0, 1]:+.0f}",
        color="#ccc",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()

print(
    "Note the two middle panels: div = 0 AND curl = 0, yet DF ≠ 0 and the circle is visibly"
)
print("deformed. Those are the components div and curl are blind to.")

# %% [markdown]
# ## 2. What is invariant: why $\operatorname{tr}$ is cheap and the antisymmetric part is expensive
#
# Change coordinates linearly, $y = Px$. A vector field transforms as $\tilde F(y) = P\,F(P^{-1}y)$,
# so its derivative is **conjugated**:
#
# $$D\tilde F \;=\; P\,(DF)\,P^{-1}.$$
#
# Now the two projections behave very differently:
#
# * $\operatorname{tr}(PJP^{-1}) = \operatorname{tr}J$ for **every** invertible $P$. The trace is a
#   $GL(n)$ invariant. Divergence therefore survives arbitrary (even non-metric) changes of frame —
#   it needs only a *volume form*, not a metric.
# * $\operatorname{skew}(PJP^{-1}) = P\,\operatorname{skew}(J)\,P^{-1}$ **only if** $P^{\mathsf T}P \propto I$.
#   The transpose is defined by the inner product, so the symmetric/antisymmetric split is a
#   *metric* notion. Curl needs a metric — and, to be a vector rather than a 2-form, an orientation
#   as well.
#
# This is not pedantry: it is the reason divergence generalizes to any manifold with a volume form
# ($\mathcal{L}_F\mu = (\operatorname{div}F)\,\mu$) while curl-as-a-vector does not.

# %%
Jrot = np.array([[0.0, -1.0], [1.0, 0.0]])  # pure rotation field (−y, x): div 0, curl 2

Pstretch = np.array([[2.0, 0.0], [0.0, 1.0]])  # non-orthogonal: an anisotropic stretch
th0 = 0.7
Q = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])  # orthogonal

for name, P in [("stretch diag(2,1)", Pstretch), ("rotation by 0.7 rad", Q)]:
    Jc = P @ Jrot @ np.linalg.inv(P)
    print(f"\nconjugating by {name}:")
    print("   J~ =\n", np.round(Jc, 4))
    print(
        f"   tr  : {np.trace(Jrot) + 0.0:+.4f} → {np.round(np.trace(Jc), 12) + 0.0:+.4f}"
        f"   (invariant: {np.isclose(np.trace(Jrot), np.trace(Jc))})"
    )
    print(f"   curl: {Jrot[1, 0] - Jrot[0, 1]:+.4f} → {Jc[1, 0] - Jc[0, 1]:+.4f}")
    print(
        "   skew(P J P⁻¹) == P skew(J) P⁻¹ :",
        np.allclose(skew(Jc), P @ skew(Jrot) @ np.linalg.inv(P)),
    )
    strained = not np.allclose(sym(Jc), 0, atol=1e-12)
    print(
        "   symmetric part now nonzero?",
        strained,
        " →  the pure rotation has picked up a strain part"
        if strained
        else " →  still a pure rotation: orthogonal changes preserve the split",
    )

# %% [markdown]
# So in the stretched coordinates the *same physical motion* acquires a strain part and a different
# curl, while the divergence is untouched. Divergence is a $GL$-invariant of $DF$; curl is only an
# $O(n)$-covariant.
#
# > **On a manifold.** For a vector field $F$ on $M$, $DF$ is only defined once you choose a
# > connection $\nabla$; then $\operatorname{div}F = \operatorname{tr}\nabla F$ and the trace kills the
# > connection-dependence for the Levi-Civita choice. Equivalently, and better, define divergence with
# > no connection at all by $\mathcal{L}_F\mu = (\operatorname{div}_\mu F)\,\mu$ for a volume form $\mu$
# > — §5 shows this is the same thing.

# %% [markdown]
# ## 3. Divergence = rate of change of volume (Liouville)
#
# Let $\varphi_t$ be the flow of $F$ and $\Phi_t(p) = D\varphi_t(p)$ the Jacobian of the flow, which
# solves the **variational equation**
#
# $$\dot\Phi_t \;=\; DF(\varphi_t(p))\,\Phi_t, \qquad \Phi_0 = I .$$
#
# Jacobi's formula, $\frac{d}{dt}\det \Phi = \det\Phi\,\operatorname{tr}(\Phi^{-1}\dot\Phi)$, then gives
# **Liouville's formula**
#
# $$\frac{d}{dt}\det\Phi_t \;=\; \operatorname{div}F(\varphi_t(p))\;\det\Phi_t
# \qquad\Longrightarrow\qquad
# \det D\varphi_t(p) \;=\; \exp\!\int_0^t \operatorname{div}F(\varphi_s(p))\,ds .$$
#
# At $t=0$ this is the clean statement
#
# $$\left.\frac{d}{dt}\right\rvert_{0}\det D\varphi_t \;=\; \operatorname{tr}DF \;=\;\operatorname{div}F,$$
#
# i.e. **divergence is the logarithmic rate at which the flow expands volume**. (The
# `GeometricLinearAlgebra/04_Volume_Determinant_Trace` notebook derives the linear case,
# $\frac{d}{dt}\big\rvert_0\det(I+tT) = \operatorname{tr}T$; this is its nonlinear form.)


# %%
def flow_with_jacobian(F, p0, T, n=4000):
    """RK4 on the augmented state (x, Φ) with Φ̇ = DF(x) Φ, Φ(0) = I.

    Returns the trajectory, the flow Jacobians along it, and ∫ div F dt.
    """
    p0 = np.asarray(p0, float)
    d = p0.size
    x, Phi, integral = p0.copy(), np.eye(d), 0.0
    dt = T / n
    xs, Phis, integrals = [x.copy()], [Phi.copy()], [0.0]

    def rhs(x, Phi):
        return np.asarray(F(x), float), jacobian(F, x) @ Phi

    for _ in range(n):
        div_before = np.trace(jacobian(F, x))
        k1x, k1P = rhs(x, Phi)
        k2x, k2P = rhs(x + 0.5 * dt * k1x, Phi + 0.5 * dt * k1P)
        k3x, k3P = rhs(x + 0.5 * dt * k2x, Phi + 0.5 * dt * k2P)
        k4x, k4P = rhs(x + dt * k3x, Phi + dt * k3P)
        x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        Phi = Phi + (dt / 6) * (k1P + 2 * k2P + 2 * k3P + k4P)
        integral += (
            0.5 * dt * (div_before + np.trace(jacobian(F, x)))
        )  # trapezoid, O(dt²)
        xs.append(x.copy())
        Phis.append(Phi.copy())
        integrals.append(integral)
    return np.array(xs), np.array(Phis), np.array(integrals)


# a compressible planar field with position-dependent divergence
def Fcomp(q):
    x, y = q
    return np.array(
        [x - x**3 / 3.0 - y, x + 0.5 * y - 0.4 * x * y + 0.2 * np.sin(3 * x)]
    )


q0 = np.array([0.6, 0.2])
T = 1.5
xs, Phis, integ = flow_with_jacobian(Fcomp, q0, T, n=3000)
ts = np.linspace(0, T, len(xs))

dets = np.array([np.linalg.det(P_) for P_ in Phis])
print("det Dφ_T (integrated variational equation) :", f"{dets[-1]:.8f}")
print("exp ∫₀ᵀ div F dt (quadrature along orbit)  :", f"{np.exp(integ[-1]):.8f}")
print(
    "Liouville, det Dφ_t = exp ∫ div F :", np.allclose(dets, np.exp(integ), rtol=1e-5)
)

# and the t = 0 statement, d/dt|₀ det Dφ_t = div F(p)
eps = 1e-5
_, Pe, _ = flow_with_jacobian(Fcomp, q0, eps, n=20)
print(
    "\nd/dt|₀ det Dφ_t = div F(p) :",
    np.isclose((np.linalg.det(Pe[-1]) - 1.0) / eps, div(Fcomp, q0), atol=1e-4),
    f"   ({(np.linalg.det(Pe[-1]) - 1.0) / eps:.6f} vs {div(Fcomp, q0):.6f})",
)

# %% [markdown]
# The same statement, done with actual area: carry a small disc of material points along the flow and
# watch its area. The area ratio must equal $\det D\varphi_t$.

# %%
th = np.linspace(0, 2 * np.pi, 400)
r0 = 0.12
blob0 = q0[:, None] + r0 * np.stack([np.cos(th), np.sin(th)])


def advect(F, pts, T, n=1500):
    """RK4-advect a set of points (2, N) for time T."""
    p = np.array(pts, float)
    dt = T / n
    for _ in range(n):
        k1 = np.stack([F(c) for c in p.T], axis=1)
        k2 = np.stack([F(c) for c in (p + 0.5 * dt * k1).T], axis=1)
        k3 = np.stack([F(c) for c in (p + 0.5 * dt * k2).T], axis=1)
        k4 = np.stack([F(c) for c in (p + dt * k3).T], axis=1)
        p = p + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


def shoelace(pts):
    x, y = pts
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


snap_ts = [0.0, 0.5, 1.0, 1.5]
blobs = [advect(Fcomp, blob0, t) if t > 0 else blob0 for t in snap_ts]
areas = np.array([shoelace(b) for b in blobs])
pred = np.array(
    [np.linalg.det(Phis[np.argmin(abs(ts - t))]) for t in snap_ts]
) * shoelace(blob0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
fig.suptitle(
    "Divergence is the logarithmic rate of volume change:  area(t) / area(0) = det Dφ_t = exp ∫ div F dt",
    color="white",
    fontsize=11,
)
gx, gy = np.meshgrid(np.linspace(-2, 2.4, 19), np.linspace(-1.6, 2.4, 19))
U = np.zeros_like(gx)
V = np.zeros_like(gy)
for i in range(gx.shape[0]):
    for j in range(gx.shape[1]):
        U[i, j], V[i, j] = Fcomp([gx[i, j], gy[i, j]])
mag = np.hypot(U, V)
ax1.quiver(gx, gy, U / mag, V / mag, mag, cmap="viridis", alpha=0.75)
for b, t, c in zip(blobs, snap_ts, [BLUE, GREEN, YELLOW, RED]):
    ax1.fill(b[0], b[1], color=c, alpha=0.35)
    ax1.plot(b[0], b[1], "-", color=c, lw=2, label=f"t = {t}")
ax1.plot(xs[:, 0], xs[:, 1], "-", color="white", lw=1.2, alpha=0.8)
ax1.set_title("a material disc carried by the flow", color="#ccc", fontsize=10)
ax1.set_aspect("equal")
ax1.legend(fontsize=8, loc="upper left")

ax2.plot(
    ts, dets, "-", color=BLUE, lw=2.5, label=r"$\det D\varphi_t$ (variational eq.)"
)
ax2.plot(
    ts,
    np.exp(integ),
    "--",
    color=ORANGE,
    lw=2,
    label=r"$\exp\int_0^t \mathrm{div}\,F\,ds$",
)
ax2.plot(
    snap_ts,
    areas / shoelace(blob0),
    "o",
    color=GREEN,
    ms=9,
    label="measured area ratio",
)
ax2.axhline(1.0, color=GREY, lw=1, ls=":")
ax2.set_xlabel("t")
ax2.set_ylabel("volume ratio")
ax2.set_title("three ways to the same curve", color="#ccc", fontsize=10)
ax2.grid(True)
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("measured area ratio  :", np.round(areas / shoelace(blob0), 5))
print("predicted det Dφ_t   :", np.round(pred / shoelace(blob0), 5))
print("agree :", np.allclose(areas, pred, rtol=5e-3))

# %% [markdown]
# **Corollary (Liouville's theorem).** $\operatorname{div}F \equiv 0 \iff$ the flow preserves volume.
# For a Hamiltonian field $F = (\partial_p H, -\partial_q H)$ the divergence is
# $\partial_q\partial_p H - \partial_p\partial_q H = 0$ identically — phase-space volume is conserved
# because divergence is a trace and the Hamiltonian Jacobian is traceless by construction.

# %%
Hq, Hp = sp.symbols("q p", real=True)
H = Hp**2 / 2 + Hq**4 / 4 - Hq**2 / 2  # a double-well Hamiltonian
XH = sp.Matrix([sp.diff(H, Hp), -sp.diff(H, Hq)])
JH = XH.jacobian([Hq, Hp])
print("Hamiltonian vector field X_H =", sp.simplify(XH.T))
print("DX_H =", sp.simplify(JH).tolist())
print(
    "div X_H = tr DX_H =", sp.simplify(JH.trace()), "  (identically zero for every H)"
)

# %% [markdown]
# ## 4. Curl = twice the angular velocity (Cauchy–Stokes)
#
# For short times $\varphi_t \approx \mathrm{id} + tF$, so $D\varphi_t \approx I + tJ$ with $J = DF$.
# Take the **polar decomposition** $D\varphi_t = R_t U_t$ ($R_t$ orthogonal, $U_t$ symmetric positive
# definite). To first order,
#
# $$R_t \;=\; I + tA + O(t^2), \qquad U_t \;=\; I + tS + O(t^2),$$
#
# because $I + tJ = I + t(S+A)$ and the symmetric/antisymmetric split *is* the linearized polar
# decomposition. So the **antisymmetric part is exactly the instantaneous rigid-body rotation rate of
# the material**, with angular velocity
#
# $$\omega \;=\; \mathrm{vee}(A) \;=\; \tfrac12\operatorname{curl}F .$$
#
# This is Cauchy's and Stokes' decomposition of the relative motion near a point (Stokes 1845, §1):
# translation + dilation + strain + rigid rotation. The factor $\tfrac12$ is *not* a convention — it is
# the price of turning a 2-form into a vector, and it is why a paddle wheel in a fluid spins at
# $\tfrac12\lvert\operatorname{curl}u\rvert$, not $\lvert\operatorname{curl}u\rvert$.

# %%
p = np.array([0.4, -0.7, 1.1])
J = jacobian(F3, p)
for t in [0.1, 0.01, 0.001]:
    Phi = expm(t * J)  # the linearized flow's Jacobian
    R, U = polar(Phi)
    # rotation vector of R, to first order:  R ≈ I + t A  ⇒  vee(skew(R − I)) / t → ω
    omega_num = vee(skew(R - np.eye(3))) / t
    print(
        f"t = {t:6.3f}:  ω from polar decomposition = {np.round(omega_num, 5)}"
        f"   ‖ω − ½curl F‖ = {np.linalg.norm(omega_num - 0.5 * curl3(F3, p)):.2e}"
    )
print("\n½ curl F(p) =", np.round(0.5 * curl3(F3, p), 5))
print("\nand the stretch U_t = I + tS + O(t²) recovers the symmetric part:")
for t in [0.1, 0.01, 0.001]:
    U = polar(expm(t * J))[1]
    print(
        f"t = {t:6.3f}:  U symmetric: {np.allclose(U, U.T)}"
        f"   ‖(U − I)/t − S‖ = {np.linalg.norm((U - np.eye(3)) / t - sym(J)):.2e}   (falls like O(t) ✓)"
    )

# %% [markdown]
# ### The paddle wheel, and why the *average* matters
#
# Take a material line element $a$ at $p$; it is carried by the flow as $\dot a = J a$. Its angular
# velocity in the plane is
#
# $$\omega(a) \;=\; \frac{a \wedge Ja}{\lvert a\rvert^{2}}
#   \;=\; \underbrace{\tfrac12\operatorname{curl}F}_{\text{same for every }a}
#      \;+\; \underbrace{S_{21}\!\left(a_1^2-a_2^2\right)+(S_{22}-S_{11})a_1a_2}_{\text{strain: depends on the direction }a}.$$
#
# The strain term averages to zero over directions. Hence
#
# $$\boxed{\ \big\langle \omega(a)\big\rangle_{a\in S^{1}} \;=\; \tfrac12\operatorname{curl}F\ }$$
#
# — **curl is twice the mean angular velocity of material line elements**, not the rotation rate of any
# particular one. The classic illustration is simple shear $u = (y, 0)$, whose streamlines are
# perfectly straight yet whose curl is $-1$: the horizontal arm of a cross does not rotate at all, the
# vertical arm tips over at rate $-1$, and the mean is $-\tfrac12$.


# %%
def omega_of_direction(J, ang):
    a = np.array([np.cos(ang), np.sin(ang)])
    Ja = J @ a
    return a[0] * Ja[1] - a[1] * Ja[0]  # a ∧ Ja, with |a| = 1


shear = np.array([[0.0, 1.0], [0.0, 0.0]])  # u = (y, 0)
angs = np.linspace(0, 2 * np.pi, 721)
om = np.array([omega_of_direction(shear, a) for a in angs])
curl_shear = shear[1, 0] - shear[0, 1]

print("simple shear u = (y, 0):")
print("  div  =", np.trace(shear), "   curl =", curl_shear)
print("  ω(horizontal arm) =", omega_of_direction(shear, 0.0))
print("  ω(vertical arm)   =", omega_of_direction(shear, np.pi / 2))
print(
    "  mean over directions =",
    np.round(om[:-1].mean(), 10),
    " == ½ curl =",
    0.5 * curl_shear,
)
print("  ⟨ω⟩ == ½ curl :", np.isclose(om[:-1].mean(), 0.5 * curl_shear))

# same check on the general 3D field, per coordinate plane
J = jacobian(F3, p)
for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
    Jp = J[np.ix_([i, j], [i, j])]
    mean_om = np.mean([omega_of_direction(Jp, a) for a in angs[:-1]])
    print(
        f"  plane ({'xyz'[i]},{'xyz'[j]}): ⟨ω⟩ = {mean_om:+.6f}   ½(curl F)_{'xyz'[k]} = {0.5 * curl3(F3, p)[k]:+.6f}"
    )

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
fig.suptitle(
    "Simple shear u = (y, 0): straight streamlines, zero divergence, and curl = −1"
    "\ncurl is TWICE the MEAN angular velocity of material line elements — no single arm reports it",
    color="white",
    fontsize=11,
)

# left: a material cross advected by the shear
gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 15), np.linspace(-1.2, 1.2, 13))
ax1.quiver(gx, gy, gy, np.zeros_like(gx), color="#555", alpha=0.9)
for t, alpha in [(0.0, 1.0), (0.4, 0.55), (0.8, 0.3)]:
    M = expm(t * shear)
    for a0, c, lab in [
        (np.array([0.8, 0.0]), RED, "horizontal arm"),
        (np.array([0.0, 0.8]), PURPLE, "vertical arm"),
    ]:
        a = M @ a0
        ax1.annotate(
            "",
            xy=a,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=2.6, alpha=alpha),
        )
        ax1.annotate(
            "",
            xy=-a,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=2.6, alpha=alpha),
        )
    th_ = np.linspace(0, 2 * np.pi, 200)
    c_ = M @ np.stack([0.8 * np.cos(th_), 0.8 * np.sin(th_)])
    ax1.plot(c_[0], c_[1], "-", color=GREEN, lw=1.6, alpha=alpha)
ax1.set_title("a material cross + circle at t = 0, 0.4, 0.8", color="#ccc", fontsize=10)
ax1.set_aspect("equal")
ax1.set_xlim(-1.6, 1.6)
ax1.set_ylim(-1.3, 1.3)

# right: ω as a function of the arm's direction
ax2.plot(
    angs,
    om,
    "-",
    color=BLUE,
    lw=2.5,
    label=r"$\omega(a)$, angular velocity of arm at angle $a$",
)
ax2.axhline(
    0.5 * curl_shear,
    color=ORANGE,
    lw=2.2,
    ls="--",
    label=r"$\frac{1}{2}\,\mathrm{curl}\,u$ = mean",
)
ax2.axhline(0.0, color=RED, lw=1.4, ls=":", label="horizontal arm: does not rotate")
ax2.axhline(-1.0, color=PURPLE, lw=1.4, ls=":", label="vertical arm: rate −1")
ax2.set_xlabel("direction of the material line element (rad)")
ax2.set_ylabel(r"$\omega$")
ax2.set_xlim(0, 2 * np.pi)
ax2.grid(True)
ax2.set_ylim(-1.15, 0.78)
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax2.set_title(
    "the strain term averages away; the spin term does not", color="#ccc", fontsize=10
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. The Spivak view: it is all $d$
#
# *Calculus on Manifolds* refuses to treat div, grad and curl as three separate operators. There is
# one operator, the exterior derivative $d$, applied in three degrees; the metric and the orientation
# are what disguise it as three.
#
# From a vector field $F$ on $\mathbb{R}^n$ build two forms:
#
# | object | built with | formula in $\mathbb{R}^3$ |
# |---|---|---|
# | $F^\flat \in \Omega^1$ | the **metric** | $F_1\,dx + F_2\,dy + F_3\,dz$ |
# | $\eta_F = \iota_F\mu \in \Omega^{n-1}$ | a **volume form** $\mu$ | $F_1\,dy\wedge dz + F_2\,dz\wedge dx + F_3\,dx\wedge dy$ |
#
# Then:
#
# $$\big(dF^\flat\big)_{ij} \;=\; \partial_i F_j - \partial_j F_i \;=\; 2A_{ji}
# \qquad\qquad
# d\,\eta_F \;=\; (\operatorname{div}F)\,\mu .$$
#
# The first says: **the exterior derivative of $F^\flat$ *is* the antisymmetric part of $DF$** (times
# two), living in $\Lambda^2 \cong \mathfrak{so}(n)$. The second says: **the exterior derivative of
# $\iota_F\mu$ *is* the divergence** — which, by Cartan's magic formula
# $\mathcal{L}_F\mu = d\iota_F\mu + \iota_F\,d\mu = d\iota_F\mu$ (since $d\mu = 0$), is exactly the
# volume statement of §3:
#
# $$\mathcal{L}_F\,\mu \;=\; (\operatorname{div}F)\,\mu .$$
#
# In $\mathbb{R}^3$, with the Hodge star $\star$ supplied by metric + orientation,
#
# $$\operatorname{curl}F \;=\; \big(\star\, d\, F^\flat\big)^{\sharp}, \qquad\qquad
#   \operatorname{div}F \;=\; \star\, d\, \star\, F^\flat ,$$
#
# and $\operatorname{grad}f = (df)^\sharp$. The chain $\Omega^0 \xrightarrow{d} \Omega^1
# \xrightarrow{d} \Omega^2 \xrightarrow{d} \Omega^3$ is the de Rham complex; $d^2 = 0$ is
# $\operatorname{curl}\operatorname{grad} = 0$ and $\operatorname{div}\operatorname{curl} = 0$, and the
# **Poincaré lemma** (Spivak Ch. 4) is their converse on a star-shaped domain.
#
# Let us implement $d$ and $\star$ from the definitions, on **generic** functions, so that a
# `simplify(...) == 0` is a theorem about *all* smooth fields, not a spot check.

# %%
X, Y, Z = sp.symbols("x y z", real=True)
CO = (X, Y, Z)

# generic components — nothing is assumed about them
f = sp.Function("f")(X, Y, Z)
Fc = [
    sp.Function("F1")(X, Y, Z),
    sp.Function("F2")(X, Y, Z),
    sp.Function("F3")(X, Y, Z),
]


def d(form):
    """Exterior derivative of a k-form {sorted index tuple: coefficient} on ℝ³.

    d(c dx^I) = Σ_i ∂_i c  dx^i ∧ dx^I, reordered to increasing indices; the sign is
    (−1)^{#{ j ∈ I : j < i }}.
    """
    out = {}
    for I, c in form.items():
        for i in range(3):
            if i in I:
                continue
            Jn = tuple(sorted(I + (i,)))
            sign = (-1) ** Jn.index(i)
            out[Jn] = sp.expand(out.get(Jn, 0) + sign * sp.diff(c, CO[i]))
    return {I: sp.simplify(c) for I, c in out.items() if sp.simplify(c) != 0}


def _perm_sign(t):
    s = 1
    for a in range(len(t)):
        for b in range(a + 1, len(t)):
            if t[a] > t[b]:
                s = -s
    return s


def star(form):
    """Euclidean Hodge star on ℝ³, orientation dx∧dy∧dz."""
    out = {}
    for I, c in form.items():
        Ic = tuple(i for i in range(3) if i not in I)
        out[Ic] = sp.simplify(out.get(Ic, 0) + _perm_sign(I + Ic) * c)
    return {I: c for I, c in out.items() if c != 0}


def flat(Fv):
    """F ↦ F♭ = Σ Fᵢ dxⁱ  (needs the metric)."""
    return {(i,): Fv[i] for i in range(3)}


def sharp(one_form):
    """(·)♯, the inverse of ♭ in an orthonormal frame."""
    return [sp.simplify(one_form.get((i,), 0)) for i in range(3)]


def show(form):
    lbl = {
        (): "1",
        (0,): "dx",
        (1,): "dy",
        (2,): "dz",
        (0, 1): "dx∧dy",
        (0, 2): "dx∧dz",
        (1, 2): "dy∧dz",
        (0, 1, 2): "dx∧dy∧dz",
    }
    return "  +  ".join(f"({sp.simplify(c)}) {lbl[I]}" for I, c in sorted(form.items()))


# --- the two headline identities, proven for generic F ---
DF = sp.Matrix(3, 3, lambda i, j: sp.diff(Fc[i], CO[j]))
A_sym = (DF - DF.T) / 2
div_sym = sp.simplify(DF.trace())
curl_sym = sp.Matrix(
    [
        sp.diff(Fc[2], Y) - sp.diff(Fc[1], Z),
        sp.diff(Fc[0], Z) - sp.diff(Fc[2], X),
        sp.diff(Fc[1], X) - sp.diff(Fc[0], Y),
    ]
)

dFflat = d(flat(Fc))
print("d(F♭) =", show(dFflat))
print(
    "\n(dF♭)_{ij} == 2 A_{ji} for all i<j :",
    all(
        sp.simplify(dFflat.get((i, j), 0) - 2 * A_sym[j, i]) == 0
        for i, j in [(0, 1), (0, 2), (1, 2)]
    ),
)

eta = star(flat(Fc))  # ι_F μ  (equals ⋆F♭ in an orthonormal frame)
print("\nι_F μ = ⋆F♭ =", show(eta))
print("d(ι_F μ) =", show(d(eta)))
print("d(ι_F μ) == (div F) dx∧dy∧dz :", sp.simplify(d(eta)[(0, 1, 2)] - div_sym) == 0)

print(
    "\n⋆d⋆F♭ == div F   :",
    sp.simplify(star(d(star(flat(Fc)))).get((), 0) - div_sym) == 0,
)
print(
    "(⋆dF♭)♯ == curl F :",
    all(sp.simplify(a - b) == 0 for a, b in zip(sharp(star(d(flat(Fc)))), curl_sym)),
)

# --- d² = 0 gives the two classical vanishing identities, for free ---
print("\nd(df) = 0            ⇒ curl grad f = 0 :", d(d({(): f})) == {})
print("d(dF♭) = 0           ⇒ div curl F  = 0 :", d(d(flat(Fc))) == {})

# %% [markdown]
# ### Stokes' theorem is the coordinate-free *definition*
#
# Because $\int_{\partial c}\omega = \int_c d\omega$ for every chain $c$ (Spivak Ch. 4–5), shrinking
# the chain to a point turns the two identities above into limits that make no reference to
# coordinates at all:
#
# $$\operatorname{div}F(p)\;=\;\lim_{V\downarrow p}\frac{1}{\lvert V\rvert}\oint_{\partial V}F\cdot n\,dA,
# \qquad
# \big(\operatorname{curl}F\cdot n\big)(p)\;=\;\lim_{\Sigma\downarrow p}\frac{1}{\lvert \Sigma\rvert}\oint_{\partial\Sigma}F\cdot dr .$$
#
# This is the physicist's definition — flux per unit volume, circulation per unit area — and it is why
# the trace and the antisymmetric part were the right things to look at: they are the two pieces of
# $DF$ that survive integration over a small sphere. (The trace-free symmetric part integrates to zero
# over any sphere, which is precisely why div and curl cannot see it.)
#
# Both limits converge like $O(r^2)$, and we can do better than checking the slope. Averaging a
# smooth $g$ over a ball of radius $r$ in $\mathbb{R}^n$ gives
# $\bar g = g(p) + \frac{r^2}{2(n+2)}\Delta g(p) + O(r^4)$, so the leading errors are *predicted*:
#
# $$\frac{\oint_{\partial\Sigma}F\cdot dr}{\lvert\Sigma\rvert} - \operatorname{curl}F
#   \;\simeq\; \frac{r^2}{8}\,\Delta(\operatorname{curl}F),
# \qquad
# \frac{\oint_{\partial V}F\cdot n\,dA}{\lvert V\rvert} - \operatorname{div}F
#   \;\simeq\; \frac{r^2}{10}\,\Delta(\operatorname{div}F).$$
#
# (Note the corollary: if $\operatorname{div}F$ is an affine function of position the flux ratio is
# *exact* at every radius — the mean-value property. Curvature is what makes this a limit at all.)


# %%
def circulation_over_area(F, p, r, n=800):
    """(1/πr²) ∮ F·dr around a circle of radius r about p, in the plane."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = p[:, None] + r * np.stack([np.cos(t), np.sin(t)])
    tang = r * np.stack([-np.sin(t), np.cos(t)])  # dr/dt
    vals = np.stack([F(c) for c in pts.T], axis=1)
    return np.sum(np.sum(vals * tang, axis=0)) * (2 * np.pi / n) / (np.pi * r**2)


def flux_over_volume(F, p, r, nphi=64, nth=32):
    """(1/(4/3 πr³)) ∮ F·n dA over a sphere of radius r about p (Gauss–Legendre × trapezoid)."""
    u, wu = np.polynomial.legendre.leggauss(nth)  # u = cos θ ∈ [−1, 1]
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    wphi = 2 * np.pi / nphi
    U, PH = np.meshgrid(u, phi, indexing="ij")
    W = np.outer(wu, np.full(nphi, wphi))
    s = np.sqrt(1 - U**2)
    nrm = np.stack([s * np.cos(PH), s * np.sin(PH), U])  # outward unit normal
    pts = p[:, None, None] + r * nrm
    vals = np.empty_like(pts)
    for i in range(pts.shape[1]):
        for j in range(pts.shape[2]):
            vals[:, i, j] = F(pts[:, i, j])
    integrand = np.sum(vals * nrm, axis=0) * r**2  # dA = r² du dφ
    return np.sum(integrand * W) / (4.0 / 3.0 * np.pi * r**3)


p3 = np.array([0.4, -0.7, 1.1])
p2 = np.array([0.6, 0.2])
rs = np.array([0.4, 0.2, 0.1, 0.05, 0.025])

curl_exact = curl2(Fcomp, p2)
div_exact = div(F3, p3)
circ = np.array([circulation_over_area(Fcomp, p2, r) for r in rs])
flux = np.array([flux_over_volume(F3, p3, r) for r in rs])

print(f"scalar curl at p = {p2}:  exact {curl_exact:.8f}")
for r, c in zip(rs, circ):
    print(
        f"   r = {r:6.4f}:  circulation/area = {c:.8f}   err = {abs(c - curl_exact):.2e}"
    )
print(f"\ndivergence at p = {p3}:  exact {div_exact:.8f}")
for r, fx in zip(rs, flux):
    print(
        f"   r = {r:6.4f}:  flux/volume      = {fx:.8f}   err = {abs(fx - div_exact):.2e}"
    )

# the error must fall like r², with the coefficient predicted above
slope_c = np.polyfit(np.log(rs), np.log(abs(circ - curl_exact)), 1)[0]
slope_f = np.polyfit(np.log(rs), np.log(abs(flux - div_exact)), 1)[0]
print(
    f"\nconvergence order:  circulation {slope_c:.2f},  flux {slope_f:.2f}   (both ≈ 2 ✓)"
)


def laplacian(g, p, h=1e-3):
    """Δg by second differences, for a scalar function of a point."""
    p = np.asarray(p, float)
    out = -2.0 * g(p) * p.size
    for i in range(p.size):
        e = np.zeros(p.size)
        e[i] = h
        out += g(p + e) + g(p - e)
    return out / h**2


lap_curl = laplacian(lambda q: curl2(Fcomp, q, h=1e-5), p2)
lap_div = laplacian(lambda q: div(F3, q, h=1e-5), p3)
print(f"\nΔ(curl F) = {lap_curl:+.4f}   ⇒ predicted error  (r²/8)·Δcurl")
print(f"Δ(div  F) = {lap_div:+.4f}   ⇒ predicted error  (r²/10)·Δdiv")
for r, c, fx in zip(rs, circ, flux):
    print(
        f"   r = {r:6.4f}:  circ err {c - curl_exact:+.3e} vs {r**2 / 8 * lap_curl:+.3e}"
        f"    flux err {fx - div_exact:+.3e} vs {r**2 / 10 * lap_div:+.3e}"
    )
print(
    "\nleading-order error coefficients match :",
    np.allclose(circ[-3:] - curl_exact, rs[-3:] ** 2 / 8 * lap_curl, rtol=0.05),
    np.allclose(flux[-3:] - div_exact, rs[-3:] ** 2 / 10 * lap_div, rtol=0.05),
)

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.loglog(
    rs,
    abs(circ - curl_exact),
    "o-",
    color=BLUE,
    lw=2,
    ms=8,
    label=f"|circ/area − curl|  (slope {slope_c:.2f})",
)
ax.loglog(
    rs,
    abs(flux - div_exact),
    "s-",
    color=ORANGE,
    lw=2,
    ms=8,
    label=f"|flux/vol − div|  (slope {slope_f:.2f})",
)
ax.loglog(rs, 0.5 * rs**2, "--", color=GREY, lw=1.5, label=r"$O(r^2)$ reference")
ax.set_xlabel("radius r of the shrinking chain")
ax.set_ylabel("error")
ax.set_title(
    "Stokes' theorem as a definition\nflux/volume → div,  circulation/area → curl",
    color="#ccc",
    fontsize=10,
)
ax.grid(True, which="both")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Why curl is a vector only when $n = 3$
#
# The antisymmetric part $A$ lives in $\Lambda^2(\mathbb{R}^n)\cong\mathfrak{so}(n)$, of dimension
# $\binom{n}{2}$. To call it a *vector field* you need
#
# $$\binom{n}{2} \;=\; n \qquad\Longleftrightarrow\qquad n = 3 .$$
#
# That is the whole story of the "vector curl". In other dimensions the object $dF^\flat$ still exists
# and is still the antisymmetric part of $DF$ — it just is not a vector:
#
# | $n$ | $\dim\Lambda^2 = \binom{n}{2}$ | what "curl" is | decomposition $n^2 = 1 + (\tfrac{n(n+1)}{2}-1) + \binom{n}{2}$ |
# |---|---|---|---|
# | 2 | 1 | a **scalar** $\partial_xF_y-\partial_yF_x$ | $4 = 1 + 2 + 1$ |
# | 3 | 3 | a **vector** (pseudo-vector) | $9 = 1 + 5 + 3$ |
# | 4 | 6 | a **bivector** — e.g. $F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu$ in electromagnetism | $16 = 1 + 9 + 6$ |
# | $n$ | $\binom{n}{2}$ | a 2-form | — |
#
# The electromagnetic field tensor is literally "the curl of the 4-potential": $F = dA$, and its six
# components are $\mathbf E$ and $\mathbf B$ — three "electric" and three "magnetic" only because
# $\binom{4}{2} = 3 + 3$. The cross product and the vector curl are $n=3$ coincidences; the 2-form is
# the invariant object.

# %%
print("dim gl(n) = 1 (dilation) + [n(n+1)/2 − 1] (shear) + n(n−1)/2 (rotation)")
for n in range(2, 8):
    a, b, c = 1, n * (n + 1) // 2 - 1, n * (n - 1) // 2
    assert a + b + c == n * n
    print(
        f"  n = {n}:  {n * n:3d} = {a} + {b:2d} + {c:2d}"
        f"   curl is a {'scalar' if c == 1 else 'vector' if c == n else f'{c}-component 2-form'}"
        f"{'   ← the only n with dim Λ² = n' if c == n and n > 2 else ''}"
    )

# in 2D the whole antisymmetric part is one number, and it IS the scalar curl
Jp = jacobian(Fcomp, p2)
print(f"\n2D: skew part =\n{np.round(skew(Jp), 6)}")
print(
    f"    its single free entry ×2 = {2 * skew(Jp)[1, 0]:.6f} == scalar curl {curl2(Fcomp, p2):.6f}"
)

# %% [markdown]
# ## 7. What div and curl do *not* see — and what they nevertheless determine
#
# ### Pointwise: almost everything
#
# The trace-free symmetric part $S_0$ — the **rate of strain** — is annihilated by both operators. In
# $\mathbb{R}^3$ that is 5 of 9 components. Two fields can have identical divergence and identical
# curl at every point and still have completely different derivatives.
#
# The cleanest example: $F(x,y) = (y, x)$. It is the gradient of $xy$, so $\operatorname{curl}F=0$; it
# is trace-free, so $\operatorname{div}F=0$; and yet $DF = \left[\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right] \ne 0$.
# It is pure strain — a saddle that stretches along $y=x$ and compresses along $y=-x$ at equal rates.
# Its potential $xy$ is harmonic, which is exactly the statement $\operatorname{div}=\operatorname{curl}=0$.

# %%
invisible = [
    (lambda q: np.array([q[1], q[0]]), "F = (y, x)   = ∇(xy)"),
    (lambda q: np.array([q[0], -q[1]]), "F = (x, −y)  = ∇((x²−y²)/2)"),
    (
        lambda q: np.array([q[0] ** 2 - q[1] ** 2, -2 * q[0] * q[1]]),
        "F = Re/Im of z̄²  (harmonic)",
    ),
]
q = np.array([0.7, -0.3])
print("fields that div and curl cannot distinguish from F ≡ 0:")
for Fi, name in invisible:
    Ji = jacobian(Fi, q)
    print(
        f"  {name:32s} div = {np.trace(Ji):+.2e}   curl = {Ji[1, 0] - Ji[0, 1]:+.2e}"
        f"   ‖DF‖_F = {np.linalg.norm(Ji):.4f}  ← nonzero!"
    )

# %% [markdown]
# ### Globally: nearly everything, by Helmholtz
#
# The pointwise blindness is repaired by *global* information. Helmholtz (1858) showed that on
# $\mathbb{R}^3$, a field decaying suitably at infinity is determined by its divergence and its curl:
#
# $$F \;=\; -\nabla\phi \;+\; \nabla\times \mathbf{A},\qquad
# \phi(x)=\frac{1}{4\pi}\!\int\!\frac{\operatorname{div}F(y)}{\lvert x-y\rvert}dy,\quad
# \mathbf{A}(x)=\frac{1}{4\pi}\!\int\!\frac{\operatorname{curl}F(y)}{\lvert x-y\rvert}dy .$$
#
# Two fields with the same div and curl differ by a field that is both curl-free and divergence-free,
# hence harmonic componentwise; with decay, Liouville's theorem for harmonic functions forces it to
# vanish. On a general compact manifold the leftover space is not zero but *finite dimensional* — it is
# the space of harmonic forms, and by the **Hodge theorem** it is isomorphic to the de Rham cohomology
# $H^k_{dR}(M)$. The gap between "pointwise blind" and "globally determined" is exactly topology.
#
# We can watch this: the two "invisible" fields above are precisely harmonic, so they *are* the
# ambiguity — and on a domain with a hole, an honest nonzero example appears.


# %%
# The vortex field: curl-free and divergence-free away from the origin, yet with circulation 2π
# around any loop enclosing it. On the punctured plane its 1-form is closed but not exact —
# H¹(ℝ²∖{0}) = ℝ. This is the Poincaré lemma failing when the domain is not star-shaped.
def vortex(q):
    x, y = q
    r2 = x * x + y * y
    return np.array([-y / r2, x / r2])


for test_p in [np.array([1.0, 0.0]), np.array([0.3, 1.4]), np.array([-2.0, 0.5])]:
    Jv = jacobian(vortex, test_p, h=1e-5)
    print(
        f"  at {np.round(test_p, 2)}:  div = {np.trace(Jv):+.2e}   curl = {Jv[1, 0] - Jv[0, 1]:+.2e}"
        f"   ‖skew‖ = {np.linalg.norm(skew(Jv)):.2e}"
    )

t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
for R in [0.5, 1.0, 2.5]:
    pts = R * np.stack([np.cos(t), np.sin(t)])
    tang = R * np.stack([-np.sin(t), np.cos(t)])
    vals = np.stack([vortex(c) for c in pts.T], axis=1)
    circ_R = np.sum(np.sum(vals * tang, axis=0)) * (2 * np.pi / len(t))
    print(f"  ∮ F·dr on the circle r = {R}:  {circ_R:.8f}   (= 2π = {2 * np.pi:.8f})")
print(
    "\ncurl ≡ 0 everywhere on the domain, yet the circulation is 2π on every enclosing loop:"
)
print(
    "the 1-form is closed but not exact. Poincaré's lemma needs a star-shaped domain."
)

# %% [markdown]
# ### The bigger picture: the linearization at a zero of $F$
#
# One more reason $DF$ deserves the attention rather than its two projections: at a zero of $F$, $DF$
# *is* the local dynamics. In the plane the classification is by trace and determinant — and only the
# trace is the divergence. Whether an equilibrium spirals or not is decided by
# $\tau^2 - 4\Delta$, a quantity involving the strain part that neither div nor curl reports.

# %%
print(
    "2D linear fields, all with div = 0 — the same divergence, entirely different dynamics:"
)
cases = [
    (np.array([[0.0, -1.0], [1.0, 0.0]]), "centre (pure rotation)"),
    (np.array([[0.0, 1.0], [1.0, 0.0]]), "saddle (pure strain)"),
    (np.array([[0.0, -2.0], [0.5, 0.0]]), "centre, elliptic orbits"),
    (np.array([[1.0, -1.0], [1.0, -1.0]]), "degenerate shear"),
]
for M, name in cases:
    tau, Delta = np.trace(M), np.linalg.det(M)
    ev = np.linalg.eigvals(M)
    print(
        f"  {name:26s} div = {tau:+.1f}  curl = {M[1, 0] - M[0, 1]:+.2f}  det = {Delta:+.2f}"
        f"  eigenvalues = {np.round(ev, 3)}"
    )
print(
    "\ndiv fixes only Σλ; the rest of the spectrum lives in the parts div and curl discard."
)

# %% [markdown]
# ## 8. Coda: Jacobians of learned flows
#
# The same decomposition is the working vocabulary for the Jacobian of a learned map, which may be of
# interest if you look at latent-space geometry.
#
# * **Continuous normalizing flows.** For $\dot z = F_\theta(z)$, the instantaneous change-of-variables
#   formula is $\frac{d}{dt}\log p(z(t)) = -\operatorname{div}F_\theta(z(t))$ — literally §3, Liouville
#   run backwards. This is the identity that makes neural ODEs tractable: you need only the *trace* of
#   the Jacobian, one of the three summands, and it can be estimated with a single matrix-vector
#   product by Hutchinson's estimator $\operatorname{tr}J = \mathbb{E}_v[v^{\mathsf T}Jv]$.
# * **The other two summands are what interpretability work is usually after.** The trace-free
#   symmetric part $S_0$ is the local anisotropic stretching — its spectrum is the local condition
#   number, the thing that shows up as the singular-value spectrum of the Jacobian and controls how a
#   metric is distorted between latent and data space. The antisymmetric part $A$ is the local frame
#   rotation — the part that moves features without changing any density or any length.
# * A residual block $x \mapsto x + \varepsilon f(x)$ has Jacobian $I + \varepsilon Df$, i.e. exactly
#   the $t\to 0$ flow of §4: dilation controls the log-volume, the strain part controls conditioning,
#   the spin part controls the frame twist.
#
# Below: the change-of-variables identity, verified against the flow Jacobian we already computed.

# %%
# log p_T(φ_T(x)) = log p_0(x) − ∫₀ᵀ div F dt   ⇔   p_T(φ_T(x)) · det Dφ_T = p_0(x)
Sigma = np.array([[0.6, 0.15], [0.15, 0.4]])
Sinv = np.linalg.inv(Sigma)
logp0 = lambda z: -0.5 * z @ Sinv @ z - 0.5 * np.log(np.linalg.det(2 * np.pi * Sigma))

rng = np.random.default_rng(0)
zs = rng.multivariate_normal(np.zeros(2), Sigma, size=6)
print("instantaneous change of variables  d/dt log p = −div F,  T = 1.5")
errs = []
for z0 in zs:
    _, Ph, ig = flow_with_jacobian(Fcomp, z0, 1.5, n=1500)
    lhs = logp0(z0) - ig[-1]  # transported log-density at φ_T(z0)
    rhs = logp0(z0) - np.log(np.linalg.det(Ph[-1]))  # via the flow Jacobian
    errs.append(abs(lhs - rhs))
    print(
        f"  z0 = {np.round(z0, 3)}   log p_T = {lhs:+.6f}  (via det Dφ_T: {rhs:+.6f})"
    )
print("\n−∫div F dt == −log det Dφ_T :", np.allclose(errs, 0, atol=2e-3))

# Hutchinson: the trace — i.e. the divergence — from matrix-vector products alone
J = jacobian(F3, p3)
rng = np.random.default_rng(1)
for m in [100, 10_000, 1_000_000]:
    v = rng.choice([-1.0, 1.0], size=(m, 3))  # Rademacher probes
    est = np.mean(np.einsum("ij,jk,ik->i", v, J, v))
    print(
        f"  Hutchinson tr J with m = {m:9,d} probes: {est:+.5f}   (exact {np.trace(J):+.5f})"
    )

# %% [markdown]
# ## Summary
#
# | question | answer | where it lives in $DF$ |
# |---|---|---|
# | What is $\operatorname{div}F$? | $\operatorname{tr}DF$ | the isotropic part $\tfrac{\operatorname{tr}J}{n}I$ |
# | What is $\operatorname{curl}F$? | $2\,\mathrm{vee}(\operatorname{skew}DF)$, i.e. $dF^\flat$ | the antisymmetric part $A\in\mathfrak{so}(n)\cong\Lambda^2$ |
# | What do they miss? | the rate of strain $S_0$ | the trace-free symmetric part — 5 of 9 components in $\mathbb{R}^3$ |
# | Why is $\operatorname{div}$ a trace? | $\left.\tfrac{d}{dt}\right\rvert_0\det D\varphi_t=\operatorname{tr}DF$ | volume rate (Liouville) |
# | Why the $\tfrac12$ in curl? | $A$ *is* the angular velocity; curl is $2A$ | mean spin of material line elements |
# | What structure does each need? | $\operatorname{div}$: a volume form. $\operatorname{curl}$: a metric (+ orientation for a vector) | trace is $GL$-invariant; transpose is not |
# | Why is curl a vector only in 3D? | $\dim\Lambda^2(\mathbb{R}^n)=\binom{n}{2}=n\iff n=3$ | $\mathfrak{so}(3)\cong\mathbb{R}^3$ |
# | What is the invariant statement? | $d\,\iota_F\mu=(\operatorname{div}F)\mu$ and $dF^\flat=2A$ | one operator $d$, two degrees |
#
# The single sentence: **$DF(p)$ is an endomorphism of the tangent space, endomorphisms decompose into
# dilation $\oplus$ strain $\oplus$ spin under the orthogonal group, and divergence and curl are the
# first and third projections** — a complete account of $DF$ only if the strain vanishes, and a
# complete account of $F$ only after adding global (topological) information.
#
# ## References
#
# **The forms picture**
#
# - M. Spivak, *Calculus on Manifolds*, Benjamin, 1965. Chapter 4 (fields, forms, the Poincaré lemma,
#   Stokes' theorem for chains) and Chapter 5 (integration on manifolds; the final section derives
#   Green's, Gauss's and Stokes's classical theorems as one theorem). Spivak's preface is explicit
#   that div/grad/curl are dimension-3 disguises of $d$.
# - H. Flanders, *Differential Forms with Applications to the Physical Sciences*, Academic Press, 1963
#   — the same programme aimed at physicists; §2.6–2.8 do the $\mathbb{R}^3$ dictionary carefully.
# - V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer, 1989 — Ch. 7 on
#   forms, §16 on Liouville's theorem; the cleanest statement of $\mathcal{L}_F\mu=(\operatorname{div}F)\mu$.
# - T. Frankel, *The Geometry of Physics*, 3rd ed., CUP, 2011 — §2.9, §4.3 for what needs a metric and
#   what does not.
# - T. Needham, *Visual Differential Geometry and Forms*, Princeton, 2021 — the picture-first treatment
#   of $d$ and of curl as a 2-form.
#
# **The originals**
#
# - J. C. Maxwell, "On the mathematical classification of physical quantities", *Proc. London Math.
#   Soc.* **s1-3** (1871), 224–233 — where the names *curl*, *convergence* and *slope* were coined, and
#   where the vector/pseudo-vector distinction is first argued.
# - G. G. Stokes, "On the theories of the internal friction of fluids in motion…", *Trans. Camb. Phil.
#   Soc.* **8** (1845), 287–319 — §1 contains the decomposition of relative motion near a point into
#   dilation, strain and rigid rotation (the Cauchy–Stokes decomposition of §4).
# - H. Helmholtz, "Über Integrale der hydrodynamischen Gleichungen, welche den Wirbelbewegungen
#   entsprechen", *J. reine angew. Math.* **55** (1858), 25–55; English translation by P. G. Tait,
#   *Phil. Mag.* **33** (1867), 485–512 — the decomposition and the vortex theorems.
# - J. Liouville, "Sur la théorie de la variation des constantes arbitraires", *J. Math. Pures Appl.*
#   **3** (1838), 342–349 — the formula of §3.
# - G. K. Batchelor, *An Introduction to Fluid Dynamics*, CUP, 1967 — §2.3, "Analysis of the relative
#   motion near a point", the modern textbook form of Stokes 1845 with the paddle-wheel reading.
#
# **The coda**
#
# - R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud, "Neural Ordinary Differential Equations",
#   *NeurIPS* 2018 — the instantaneous change-of-variables formula $\frac{d\log p}{dt}=-\operatorname{tr}\frac{\partial f}{\partial z}$.
# - W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, D. Duvenaud, "FFJORD: Free-form
#   Continuous Dynamics for Scalable Reversible Generative Models", *ICLR* 2019 — Hutchinson's
#   estimator applied to that trace.
# - M. F. Hutchinson, "A stochastic estimator of the trace of the influence matrix for Laplacian
#   smoothing splines", *Comm. Statist. Simulation Comput.* **18** (1989), 1059–1076.
#
# **In this repository**
#
# - `GeometricLinearAlgebra/04_Volume_Determinant_Trace.py` — the linear case:
#   $\operatorname{tr}T=\frac{d}{dt}\big\rvert_0\det(I+tT)$ and $\det e^{tT}=e^{t\operatorname{tr}T}$.
# - `LieGroups/VectorField_View_Lie_Theory.py` — flows, the Jacobi–Lie bracket, and $\exp$ as a flow.
# - `LieGroups/SO3_Lie_Theory.py` — the hat map $\mathbb{R}^3\to\mathfrak{so}(3)$ used here to turn $A$
#   into $\tfrac12\operatorname{curl}F$.
