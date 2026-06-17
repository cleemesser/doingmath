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
# # The Vector Field View of Lie Groups & Lie Algebras
#
# The other notebooks treat a Lie algebra as the **tangent space at the identity** — a set of
# matrices you exponentiate. This notebook develops the deeper, geometric picture: a Lie algebra
# element *is a vector field*, and the whole theory is a statement about **flows**.
#
# There are three equivalent faces of "what $\mathfrak{g}$ is", and we connect them:
# 1. **Tangent vectors at $e$** — the algebraic view used elsewhere.
# 2. **Infinitesimal generators of an action** — $X$ assigns to each point $p$ the velocity
#    $\xi_X(p) = \tfrac{d}{dt}\big|_0 e^{tX}\!\cdot p$; the integral curves of this field are the
#    group orbits, and $e^{tX}$ is "flow the field for time $t$."
# 3. **Left-invariant vector fields on the group manifold** — the intrinsic definition.
#
# And the punchline: the **Lie bracket is the commutator of flows** — the infinitesimal failure of
# two flows to commute is measured by the Jacobi–Lie bracket of their vector fields.

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
# ## Numerical toolkit: fields, flows, and the Jacobi–Lie bracket
#
# A **vector field** is a function $\xi : \mathbb{R}^n \to \mathbb{R}^n$ giving a velocity at each
# point. Its **flow** $\Phi^\xi_t$ moves a point along the field for time $t$ (we integrate with
# RK4). The **Jacobi–Lie bracket** of two fields is the coordinate formula
# $[\xi,\eta] = (D\eta)\,\xi - (D\xi)\,\eta$, which we evaluate with finite-difference Jacobians —
# no special structure assumed.


# %%
def num_jacobian(f, p, h=1e-6):
    """Finite-difference Jacobian (n×n) of a vector field f at point p."""
    p = np.asarray(p, dtype=float)
    n = p.size
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (np.asarray(f(p + e)) - np.asarray(f(p - e))) / (2 * h)
    return J


def jacobi_lie(f, g, p):
    """Jacobi–Lie bracket [f, g](p) = (Dg)·f − (Df)·g of two vector fields."""
    p = np.asarray(p, dtype=float)
    return num_jacobian(g, p) @ np.asarray(f(p)) - num_jacobian(f, p) @ np.asarray(g(p))


def flow(field, p0, t, n=600):
    """RK4 integration of `field` from p0 for time t — the flow Φ^field_t(p0)."""
    p = np.asarray(p0, dtype=float)
    dt = t / n
    for _ in range(n):
        k1 = np.asarray(field(p))
        k2 = np.asarray(field(p + 0.5 * dt * k1))
        k3 = np.asarray(field(p + 0.5 * dt * k2))
        k4 = np.asarray(field(p + dt * k3))
        p = p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


# %% [markdown]
# ## 1. An Algebra Element Generates a Vector Field
#
# Take SE(2) acting on the plane by $p \mapsto e^{tX} p$ (homogeneous coordinates
# $p = (x, y, 1)$). Differentiating at $t = 0$, the **fundamental vector field** of $X$ is
#
# $$\xi_X(p) = \left.\frac{d}{dt}\right|_{0} e^{tX} p = X\,p.$$
#
# The three $\mathfrak{se}(2)$ generators become three recognizable fields on the plane:
# **rotation** $e_1 \to (-y, x)$ (circulation), and the two **translations** $e_2 \to (1,0)$,
# $e_3 \to (0,1)$ (constant). The integral curves of $\xi_X$ are precisely the orbits
# $t \mapsto e^{tX} p_0$ — so the exponential map is "flow the field for unit time."

# %%
# se(2) generators (3x3 homogeneous)
e1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)  # rotation
e2 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)  # x-translation
e3 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)  # y-translation


def se2_field(X):
    """Fundamental vector field on the plane: ξ_X(x,y) = (X · (x,y,1))[:2]."""
    return lambda p: (X @ np.array([p[0], p[1], 1.0]))[:2]


# (a) the field really is the velocity of the action, and the flow is exp(tX)·p
X = 0.7 * e1 + 0.5 * e2  # a screw generator
p0 = np.array([1.2, -0.4])
analytic_velocity = (X @ np.array([*p0, 1.0]))[:2]
numeric_velocity = (expm(1e-5 * X) @ np.array([*p0, 1.0]))[:2]
numeric_velocity = (numeric_velocity - p0) / 1e-5
print(
    "ξ_X(p) == d/dt[exp(tX)p]|₀ :",
    np.allclose(analytic_velocity, numeric_velocity, atol=1e-4),
)

# (b) RK4-integrating the field reproduces the group orbit exp(tX)·p0
T = 2.0
endpoint_flow = flow(se2_field(X), p0, T)
endpoint_exp = (expm(T * X) @ np.array([*p0, 1.0]))[:2]
print(
    "flow of ξ_X for time T == exp(TX)·p₀ :",
    np.allclose(endpoint_flow, endpoint_exp, atol=1e-6),
)
print(
    "  flow endpoint =",
    np.round(endpoint_flow, 5),
    "   exp endpoint =",
    np.round(endpoint_exp, 5),
)

# --- Visualize the three generators as fields, with orbits exp(tX)·p ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.suptitle(
    "se(2) generators as vector fields on the plane; integral curves = orbits exp(tX)·p",
    color="white",
    fontsize=12,
)
gx, gy = np.meshgrid(np.linspace(-2, 2, 17), np.linspace(-2, 2, 17))
for ax, (Xg, title, color) in zip(
    axes,
    [
        (e1, "e₁ rotation → (−y, x)", BLUE),
        (e2, "e₂ x-translation → (1, 0)", ORANGE),
        (0.5 * e1 + e2, "0.5·e₁ + e₂ → screw (spiral-ish)", GREEN),
    ],
):
    f = se2_field(Xg)
    U = np.zeros_like(gx)
    Vv = np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            vel = f([gx[i, j], gy[i, j]])
            U[i, j], Vv[i, j] = vel[0], vel[1]
    ax.quiver(gx, gy, U, Vv, color="#666", alpha=0.8)
    for start in [np.array([1.0, 0.0]), np.array([0.0, 1.3]), np.array([-1.2, -0.6])]:
        ts = np.linspace(0, 3.0, 120)
        curve = np.array([(expm(t * Xg) @ np.array([*start, 1.0]))[:2] for t in ts])
        ax.plot(curve[:, 0], curve[:, 1], "-", color=color, lw=2)
        ax.plot(*start, "o", color="white", ms=5, zorder=5)
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The Lie Bracket is the Commutator of Flows
#
# Here is the geometric meaning of the bracket. Flow along $\xi_X$ for time $\varepsilon$, then
# along $\xi_Y$, then *back* along $\xi_X$, then *back* along $\xi_Y$. You do **not** return to the
# start — the small "quadrilateral" fails to close, and the closing defect is
#
# $$\Phi^{\xi_Y}_{-\varepsilon}\,\Phi^{\xi_X}_{-\varepsilon}\,\Phi^{\xi_Y}_{\varepsilon}\,\Phi^{\xi_X}_{\varepsilon}(p)
#   \;-\; p \;=\; \varepsilon^2\,[\xi_X, \xi_Y](p) \;+\; O(\varepsilon^3)$$
#
# (reading right-to-left: flow $X$ for $+\varepsilon$, then $Y$, then $X$ for $-\varepsilon$, then $Y$).
#
# We verify two things numerically: (i) the closing defect, divided by $\varepsilon^2$, converges
# to the Jacobi–Lie bracket $[\xi_X, \xi_Y]$; and (ii) for fundamental fields of an action this
# bracket equals $-\,\xi_{[X,Y]}$ — the field map $X \mapsto \xi_X$ is an **anti-homomorphism**
# (a sign flip relative to the algebra bracket, because the action multiplies on the left).
#
# > **Same idea, two faces.** The SE(2), SO(3), and SE(3) notebooks each have a "Non-commutativity
# > and the Bracket" section showing the **group commutator** — the gap between $e^{sX}e^{tY}$ and
# > $e^{tY}e^{sX}$ — closing to $st\,[X,Y]$ at the algebra level (a clean $+[X,Y]$). That is the same
# > bracket seen *on the group*; here we see it *on the acted-on space* as the **flow commutator**
# > (the non-closing quadrilateral below). Group commutator and flow commutator are two faces of the
# > one coin — they differ only by which side the group multiplies on, hence the sign in (ii).

# %%
X, Y = e1, e2  # rotation and x-translation;  algebra bracket [e1,e2] = e3
xiX, xiY = se2_field(X), se2_field(Y)
p = np.array([0.7, 0.5])


def closing_defect(eps):
    q = flow(xiX, p, eps)  # flow along X for +ε
    q = flow(xiY, q, eps)  # then along Y for +ε
    q = flow(xiX, q, -eps)  # then back along X
    q = flow(xiY, q, -eps)  # then back along Y
    return q - p


print("Closing defect of the flow quadrilateral → Jacobi–Lie bracket [ξ_X, ξ_Y](p):")
jl = jacobi_lie(xiX, xiY, p)
for eps in [0.2, 0.1, 0.05, 0.02]:
    print(
        f"  ε = {eps:4.2f}:  defect/ε² = {np.round(closing_defect(eps) / eps**2, 4)}   (Jacobi–Lie = {np.round(jl, 4)})"
    )

# the anti-homomorphism: [ξ_X, ξ_Y] equals −ξ_{[X,Y]} = ξ_{[Y,X]}
bracket_alg = X @ Y - Y @ X  # = e3
field_of_neg_bracket = se2_field(-bracket_alg)(p)
print("\nJacobi–Lie [ξ_X,ξ_Y](p) =", np.round(jl, 6))
print("−ξ_{[X,Y]}(p)            =", np.round(field_of_neg_bracket, 6))
print(
    "anti-homomorphism  [ξ_X,ξ_Y] = −ξ_{[X,Y]} :",
    np.allclose(jl, field_of_neg_bracket, atol=1e-4),
)

# --- Visualize the non-closing quadrilateral ---
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.set_title(
    "Lie bracket as a non-closing flow quadrilateral\nflow X, then Y, then −X, then −Y — the gap ∝ [ξ_X, ξ_Y]",
    color="#ccc",
    fontsize=10,
)
ax.set_aspect("equal")
ax.grid(True)
eps = 0.9
legs = [
    (xiX, eps, BLUE, "flow X (+ε)"),
    (xiY, eps, ORANGE, "flow Y (+ε)"),
    (xiX, -eps, BLUE, "flow X (−ε)"),
    (xiY, -eps, ORANGE, "flow Y (−ε)"),
]
cur = p.copy()
ax.plot(*cur, "o", color="white", ms=9, zorder=6)
ax.text(cur[0] + 0.02, cur[1] + 0.04, "start", color="white", fontsize=9)
for fld, dist, col, lab in legs:
    seg = np.array([flow(fld, cur, s) for s in np.linspace(0, dist, 40)])
    ax.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=2.5, label=lab)
    cur = seg[-1]
ax.plot(*cur, "*", color=GREEN, ms=18, zorder=6)
ax.annotate(
    "",
    xy=cur,
    xytext=p,
    arrowprops=dict(
        arrowstyle="->", color=GREEN, lw=2, connectionstyle="arc3,rad=0.25"
    ),
)
ax.text(
    (p[0] + cur[0]) / 2 - 0.15,
    (p[1] + cur[1]) / 2 + 0.12,
    "gap ∝ [ξ_X, ξ_Y]",
    color=GREEN,
    fontsize=9,
)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Left-Invariant Vector Fields: the Intrinsic Definition
#
# The most intrinsic answer to "what is $\mathfrak{g}$?" needs no action at all: $\mathfrak{g}$ is
# the space of **left-invariant vector fields** on the group manifold $G$. A tangent vector $X$ at
# the identity is spread to every point $g$ by left translation, giving the field $X^L(g) = g\,X$
# (matrix product). Two facts make this *the* definition:
#
# - The integral curve of $X^L$ through the identity is the one-parameter subgroup $t \mapsto e^{tX}$.
# - The Jacobi–Lie bracket of left-invariant fields reproduces the algebra bracket **with no sign
#   flip**: $[X^L, Y^L] = [X, Y]^L$ (a genuine homomorphism — contrast §2's action fields).
#
# We make this concrete on the smallest non-abelian Lie group, the **"$ax+b$" group** $\mathrm{Aff}^+(1)$
# of maps $x \mapsto ax + b$ with $a > 0$. Its manifold is the **upper half-plane**
# $\{(a, b) : a > 0\}$, so we can actually *draw* the fields.


# %%
def aff(a, b):
    """Element of the ax+b group as a 2x2 matrix [[a, b],[0,1]] (a > 0)."""
    return np.array([[a, b], [0, 1.0]])


# algebra basis: S = scaling generator, Tg = translation generator
S = np.array([[1.0, 0.0], [0.0, 0.0]])
Tg = np.array([[0.0, 1.0], [0.0, 0.0]])


def left_inv_field(Xalg):
    """Left-invariant field on the (a,b) upper half-plane: at (a,b), X^L(g) = g·X read off in coords."""

    def fld(p):
        a, b = p
        M = (
            aff(a, b) @ Xalg
        )  # = [[a·α, a·β],[0,0]];  tangent (da, db) = (M[0,0], M[0,1])
        return np.array([M[0, 0], M[0, 1]])

    return fld


# (a) integral curve of X^L through the identity == one-parameter subgroup exp(tX)
for Xalg, name in [(S, "scaling S"), (0.6 * S + Tg, "mixed 0.6S+T")]:
    T = 1.3
    g_flow = flow(left_inv_field(Xalg), [1.0, 0.0], T)  # start at identity (a,b)=(1,0)
    g_exp = expm(T * Xalg)
    print(
        f"flow of {name:12s} from e for time T  →  (a,b) = {np.round(g_flow, 5)};   exp(TX) → (a,b) = {np.round([g_exp[0, 0], g_exp[0, 1]], 5)};   match: {np.allclose(g_flow, [g_exp[0, 0], g_exp[0, 1]], atol=1e-6)}"
    )

# (b) Jacobi–Lie bracket of left-invariant fields == field of the ALGEBRA bracket (homomorphism)
p = np.array([1.7, -0.6])  # an arbitrary point of the half-plane
jl = jacobi_lie(left_inv_field(S), left_inv_field(Tg), p)
bracket_ST = S @ Tg - Tg @ S  # algebra bracket [S, T] = T
field_bracket = left_inv_field(bracket_ST)(p)
print("\n[S, T] (algebra) = T :", np.allclose(bracket_ST, Tg))
print("Jacobi–Lie [Sᴸ, Tᴸ](p) =", np.round(jl, 6))
print("field of [S,T]ᴸ (p)    =", np.round(field_bracket, 6))
print(
    "homomorphism  [Sᴸ,Tᴸ] = [S,T]ᴸ  (NO sign flip):",
    np.allclose(jl, field_bracket, atol=1e-4),
)

# --- Visualize left-invariant fields and their flows on the upper half-plane ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
fig.suptitle(
    "Left-invariant vector fields on the ax+b group (upper half-plane a>0)\nintegral curve through e=(1,0) is the 1-parameter subgroup exp(tX)",
    color="white",
    fontsize=11,
)
ga, gb = np.meshgrid(np.linspace(-2, 2, 17), np.linspace(0.25, 3, 14))
for ax, (Xalg, title, color) in zip(
    axes,
    [
        (S, "Sᴸ scaling generator: X^L = (a, 0)", BLUE),
        (Tg, "Tᴸ translation generator: X^L = (0, a)", ORANGE),
    ],
):
    fld = left_inv_field(Xalg)
    U = np.zeros_like(ga)
    Vv = np.zeros_like(gb)
    for i in range(ga.shape[0]):
        for j in range(ga.shape[1]):
            vel = fld([gb[i, j], ga[i, j]])  # note: x-axis=b, y-axis=a
            U[i, j], Vv[i, j] = (
                vel[1],
                vel[0],
            )  # (db, da) for plotting (b horizontal, a vertical)
    ax.quiver(gb, ga, U, Vv, color="#666", alpha=0.8)
    # integral curves through identity and two other base points (left-invariance: same field, shifted)
    for start in [[1.0, 0.0], [1.5, -1.2], [0.6, 1.0]]:  # (a, b)
        ts = np.linspace(-1.2, 1.2, 120)
        curve = np.array(
            [expm(t * Xalg) @ aff(*start) for t in ts]
        )  # left-translate? show subgroup orbit
        ab = np.array([[c[0, 0], c[0, 1]] for c in curve])
        ax.plot(ab[:, 1], ab[:, 0], "-", color=color, lw=2)
        ax.plot(start[1], start[0], "o", color="white", ms=6, zorder=5)
    ax.text(0.02, 0.0 + 1.0, "  e=(a=1,b=0)", color=GREEN, fontsize=8)
    ax.plot(0, 1, "o", color=GREEN, ms=9, zorder=6)
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_xlabel("b")
    ax.set_ylabel("a  (>0)")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(0, 3.1)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary: three views of one Lie algebra
#
# | View | $X \in \mathfrak{g}$ is… | the bracket is… | $\exp$ is… |
# |---|---|---|---|
# | **Algebraic** | a tangent vector at $e$ | matrix commutator $XY - YX$ | matrix exponential |
# | **Action / generator** | a field $\xi_X(p) = Xp$ on the acted-on space | $-[\,\xi_X,\xi_Y\,]$ *(anti-hom.)* | flow the field for time $t$ |
# | **Left-invariant** | a field $X^L(g) = gX$ on $G$ itself | $[\,X^L, Y^L\,] = [X,Y]^L$ *(hom.)* | flow from $e$ for time $t$ |
#
# The unifying ideas:
# - **$\exp$ is a flow.** A one-parameter subgroup $e^{tX}$ is the integral curve of a vector field;
#   exponentiating a matrix and following a flow for unit time are the same operation.
# - **The bracket measures non-commuting flows.** Following $X$ then $Y$ versus $Y$ then $X$ leaves a
#   second-order gap proportional to $[\xi_X, \xi_Y]$ — this is what makes a curved group "curved".
# - **Watch the sign.** Fundamental fields of a *left* action give an anti-homomorphism; the
#   intrinsic *left-invariant* fields give a true homomorphism. Both are faithful pictures of the
#   same abstract bracket — they differ only by which side the group multiplies on.
#
# This is the bridge from the matrix bookkeeping of the other notebooks to differential geometry:
# a Lie group is a manifold whose symmetry is encoded entirely in a finite-dimensional space of
# vector fields, and the Lie algebra is the calculus of their flows.
