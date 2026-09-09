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
# Any square matrix splits into a symmetric and an antisymmetric part, and the symmetric part splits
# again into a multiple of the identity plus a trace-free remainder. In words, that gives three pieces:
#
# $$DF \;=\; (\text{isotropic scaling}) \;+\; (\text{trace-free symmetric part}) \;+\; (\text{antisymmetric part})$$
#
# and explicitly:
#
# $$DF \;=\; \underbrace{\tfrac{1}{n}(\operatorname{tr} DF)\,I}_{\substack{\text{isotropic scaling}\\ \text{(uniform dilation)}}}
#            \;+\; \underbrace{S_0}_{\substack{\text{trace-free symmetric}\\ \text{(anisotropic stretch)}}}
#            \;+\; \underbrace{A}_{\substack{\text{antisymmetric}\\ \text{(infinitesimal rotation)}}}$$
#
# The word *isotropic* is carrying real weight there. $\tfrac{1}{n}(\operatorname{tr}DF)I$ is not "the
# diagonal of $DF$"; it is the diagonal's **average**, copied into every diagonal slot. Any way in
# which the stretching is direction-dependent is a deviation from that average, and those deviations
# live in $S_0$, whose own diagonal is nonzero in general. Scaling is therefore split across the first
# *and* second pieces --- only the isotropic part of it is in the first.
#
# **An example in $\mathbb{R}^3$.** Take $F(x,y,z) = (3x,\,0,\,0)$ --- material pulled along the
# $x$-axis and left alone in $y$ and $z$. Its derivative is constant, and splits as
#
# $$DF \;=\; \begin{pmatrix}3&0&0\\0&0&0\\0&0&0\end{pmatrix}
# \;=\; \underbrace{\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}}_{\tfrac{1}{3}(\operatorname{tr}DF)\,I}
# \;+\; \underbrace{\begin{pmatrix}2&0&0\\0&-1&0\\0&0&-1\end{pmatrix}}_{S_0}
# \;+\; \underbrace{\vphantom{\begin{pmatrix}1\\1\\1\end{pmatrix}}0}_{A}$$
#
# Read the two nonzero pieces: the first says *expand uniformly at rate $1$ in every direction*, the
# second says *and on top of that, stretch at rate $2$ along $x$ while contracting at rate $1$ in $y$
# and $z$*. Neither is a description of $F$; only the sum is. Notice in particular that the first
# piece assigns motion to $y$ and $z$, where $F$ moves nothing at all --- the average of $(3,0,0)$
# knows nothing about which direction was the stretched one.
#
# Now compare $G(x,y,z) = (x,y,z)$, a uniform radial expansion. Both fields have
# $\operatorname{div} = 3$ and $\operatorname{curl} = 0$, so div and curl report them as the same
# field. Both multiply volume at the identical rate $e^{3t}$ along their flows --- that *is* what the
# divergence measures. But $G$ inflates a ball into a larger ball while $F$ draws it out into a cigar,
# and the whole of that difference is $S_0 = \operatorname{diag}(2,-1,-1)$, the piece neither operator
# sees.
#
# and the claim of this notebook is that
#
# > **divergence is the first piece and curl is the third**, and *the middle piece is invisible to
# > both*.
#
# In $\mathbb{R}^3$ that middle piece has 5 of the 9 components. So "div and curl" is not a complete
# description of $DF$ — it reads two of its three parts and discards the largest one. Knowing this
# tells you exactly what div and curl can and cannot see, why curl is a *vector* only in three
# dimensions, and why div needs less structure than curl. **Gradient** joins the story from the other
# end: it is the operator that *builds* vector fields whose third piece is zero.
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
# **Prerequisites.** Sections 1–9 assume only a first undergraduate course in linear algebra
# (matrices, transpose, trace, eigenvalues, the dot product) and multivariable calculus. Section 10
# is a closing note for readers who want the group-theoretic statement; nothing before it depends on
# that language.
#
# **Contents**
#
# 1. The decomposition of $DF$ and the two projections
# 2. Gradient: the operator that *makes* fields with no rotation part
# 3. What is invariant: why $\operatorname{tr}$ is cheap and the antisymmetric part is expensive
# 4. Divergence = rate of change of volume (Liouville)
# 5. Curl = twice the angular velocity (Cauchy–Stokes)
# 6. The Spivak view: it is all $d$
# 7. Why curl is a vector only when $n = 3$
# 8. What div and curl do *not* see — and what they nevertheless determine
# 9. A coda: Jacobians of learned flows
# 10. Advanced note: why these three pieces, and why *only* div and curl

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
# A finite-difference Jacobian, the three projections, and a pair of maps we will lean on throughout.
#
# **The hat map $\widehat{\;\cdot\;}$ and the vee map $\mathrm{vee}$.** An antisymmetric $3\times3$
# matrix has zeros on the diagonal and only three independent entries, so it carries exactly as much
# information as a vector in $\mathbb{R}^3$. The two maps that translate between them are
#
# $$\widehat{w}\;=\;\begin{pmatrix} 0 & -w_3 & w_2\\ w_3 & 0 & -w_1\\ -w_2 & w_1 & 0\end{pmatrix},
# \qquad\qquad
# \mathrm{vee}\!\begin{pmatrix} 0 & -c & b\\ c & 0 & -a\\ -b & a & 0\end{pmatrix} \;=\; (a,\,b,\,c).$$
#
# In words: **hat** takes a vector and builds the antisymmetric matrix whose action is "cross with
# that vector", $\widehat{w}\,v = w\times v$; **vee** is its inverse, reading the three independent
# entries $\mathrm{vee}(A)=(A_{32},\,A_{13},\,A_{21})$ back off as a vector. They are mutually
# inverse bijections — $\mathrm{vee}(\widehat{w})=w$, and $\widehat{\mathrm{vee}(A)}=A$ for every
# antisymmetric $A$ — so "antisymmetric $3\times3$ matrix" and "vector in $\mathbb{R}^3$" are two
# spellings of the same thing. That bijection is the entire reason a curl can be written as a vector
# in three dimensions, and §7 shows it exists in no other dimension. (Same convention as the
# `LieGroups/SO3_Lie_Theory` notebook.)


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
    """ℝ³ → antisymmetric 3×3 matrices:  hat(w) v = w × v."""
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def vee(A):
    """Antisymmetric 3×3 matrices → ℝ³, the inverse of hat."""
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


# hat and vee are mutually inverse bijections between ℝ³ and the antisymmetric 3×3 matrices
_w = np.array([0.3, -1.2, 0.8])
_v = np.array([1.0, 2.0, -0.5])
_A = (
    np.arange(9).reshape(3, 3) - np.arange(9).reshape(3, 3).T
)  # some antisymmetric matrix
print("hat(w) is antisymmetric    :", np.allclose(hat(_w), -hat(_w).T))
print("hat(w) v == w × v          :", np.allclose(hat(_w) @ _v, np.cross(_w, _v)))
print("vee(hat(w)) == w           :", np.allclose(vee(hat(_w)), _w))
print("hat(vee(A)) == A  (A skew) :", np.allclose(hat(vee(_A)), _A))


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
# is just linear algebra: every matrix is (symmetric part) + (antisymmetric part), and every symmetric
# matrix is (multiple of $I$) + (trace-free symmetric). Counting dimensions,
#
# $$\underbrace{n^2}_{\text{all matrices}} \;=\; \underbrace{1}_{\text{dilation}}
#   \;+\; \underbrace{\tfrac{n(n+1)}{2}-1}_{\text{trace-free symmetric}}
#   \;+\; \underbrace{\tfrac{n(n-1)}{2}}_{\text{antisymmetric}} .$$
#
# In $n=3$ that reads $9 = 1 + 5 + 3$: one number, five, and three. Physicists will recognize the
# same $1+5+3$ from reducing a rank-2 Cartesian tensor; §10 says why the split is forced rather than
# merely convenient.
#
# The two classical operators are the two "easy" summands:
#
# $$\boxed{\operatorname{div}F \;=\; \operatorname{tr}DF} \qquad\qquad
#   \boxed{A\,v \;=\; \tfrac12\,(\operatorname{curl}F)\times v \quad\text{in } \mathbb{R}^3}$$
#
# i.e. $\operatorname{curl}F = 2\,\mathrm{vee}(A)$. Divergence *is* the trace; curl *is* (twice) the
# antisymmetric part, repackaged as a vector by the hat map. Nothing else about $DF$ appears.

# %% [markdown]
# ### What the first piece is *not*
#
# It is tempting to read $\tfrac{\operatorname{tr}J}{n}I$ as "the diagonal of $J$" and $S_0$ as "the
# off-diagonal part". Both halves of that are wrong, and the error matters here because it would make
# divergence look like a complete account of the stretching.
#
# In the plane, $\operatorname{diag}(2,0)$ --- stretch $x$, leave $y$ alone --- and
# $\operatorname{diag}(1,1)$ --- expand uniformly in every direction --- have the **same** divergence
# $2$. What separates them is $S_0 = \operatorname{diag}(1,-1)$: diagonal, traceless, and invisible to
# div and curl alike. Non-isotropic scaling is not in the first piece at all.
#
# And "diagonal versus off-diagonal" is not a property of the map, only of the basis it was written
# in. Rotating $\operatorname{diag}(1,-1)$ by $45^\circ$ gives
# $\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$ --- the same deformation viewed in a
# turned frame. That is precisely why the splitting below is by isotropic / trace-free /
# antisymmetric, which rotations preserve, rather than by where the entries happen to sit.

# %%
print("same div, different geometry -- the first piece cannot tell these apart:")
for Jd, name in (
    (np.diag([1.0, 1.0]), "diag( 1, 1)  uniform expansion"),
    (np.diag([2.0, 0.0]), "diag( 2, 0)  stretch x only"),
    (np.diag([3.0, -1.0]), "diag( 3,-1)  stretch x, squeeze y"),
):
    print(
        f"  {name:34s} div = {np.trace(Jd):+.0f}"
        f"   iso diag = {np.diag(iso(Jd))}   S0 diag = {np.diag(dev(Jd))}"
    )

# S0 carries a diagonal of its own: the deviations of diag J from its average
Janis = np.array([[3.0, 1.0], [0.5, -1.0]])
print("\nJ =", Janis.tolist())
print("  diag J   :", np.diag(Janis))
print("  iso diag :", np.diag(iso(Janis)), "  <- the average of diag J, repeated")
print("  S0  diag :", np.diag(dev(Janis)), "  <- the deviations from that average")
assert np.allclose(np.diag(iso(Janis)) + np.diag(dev(Janis)), np.diag(Janis))

# the R^3 example from the introduction: a uniaxial stretch is not a pure dilation
J3 = np.diag([3.0, 0.0, 0.0])  # F(x,y,z) = (3x, 0, 0)
G3 = np.eye(3)  # G(x,y,z) = (x,  y,  z)
print("\nR^3:  F = (3x,0,0)   vs   G = (x,y,z)")
print(f"  div F = {np.trace(J3):+.0f}   div G = {np.trace(G3):+.0f}   -> identical")
print("  S0 of F diag :", np.diag(dev(J3)), "  S0 of G diag :", np.diag(dev(G3)))
assert np.allclose(iso(J3), np.eye(3))
assert np.allclose(dev(J3), np.diag([2.0, -1.0, -1.0]))
assert np.allclose(skew(J3), 0.0)
assert np.isclose(np.trace(J3), np.trace(G3))  # same div, different geometry

# and 'diagonal' is basis-dependent: 45 degrees turns one strain atom into the other
_th = np.pi / 4
R45 = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
print(
    "\nR(45) diag(1,-1) R(45)^T =\n", np.round(R45 @ np.diag([1.0, -1.0]) @ R45.T, 12)
)
assert np.allclose(
    R45 @ np.diag([1.0, -1.0]) @ R45.T, np.array([[0.0, 1.0], [1.0, 0.0]])
)


# %% [markdown]
# ### The general case, symbolically
#
# The uniaxial example was one matrix; the same accounting can be made once and for all. Write a
# completely general Jacobian and let $m$ be the mean of its diagonal:
#
# $$DF = \begin{pmatrix}a&b&c\\d&e&f\\g&h&i\end{pmatrix},
#   \qquad m \;=\; \tfrac{1}{3}\operatorname{tr}DF \;=\; \tfrac{a+e+i}{3}.$$
#
# Define $\sigma_1,\sigma_2,\sigma_3$ --- the diagonal of $S_0$ --- as the departure of each diagonal
# entry from that mean:
#
# $$\sigma_1 = a - m = \tfrac{2a-e-i}{3}, \qquad
#   \sigma_2 = e - m = \tfrac{2e-a-i}{3}, \qquad
#   \sigma_3 = i - m = \tfrac{2i-a-e}{3}.$$
#
# The diagonal of $DF$ is then $(m+\sigma_1,\; m+\sigma_2,\; m+\sigma_3)$, and the $\sigma_k$ obey
#
# $$\sigma_1 + \sigma_2 + \sigma_3 = 0$$
#
# identically, so they carry **two** independent numbers, not three.
#
# That is the whole accounting, and it settles the question this subsection opened with. The diagonal
# of $DF$ holds three degrees of freedom. The divergence captures exactly **one** of them --- the mean
# $m$ --- and the other two are the $\sigma_k$, sitting in $S_0$ where neither div nor curl reaches.
# The off-diagonal entries split the same way: their symmetric halves $\tfrac{b+d}{2}$,
# $\tfrac{c+g}{2}$, $\tfrac{f+h}{2}$ also join $S_0$, giving it $2 + 3 = 5$ components --- the $5$ in
# $9 = 1 + 5 + 3$ --- while their antisymmetric halves are the curl.

# %%
# Generic entries, so the identities below are proofs rather than spot checks.
syms = sp.symbols("a b c d e f g h i", real=True)
Jg = sp.Matrix(3, 3, syms)
m_g = sp.Rational(1, 3) * Jg.trace()

iso_g = m_g * sp.eye(3)
S0_g = sp.simplify((Jg + Jg.T) / 2 - iso_g)
A_g = sp.simplify((Jg - Jg.T) / 2)
sigma = [sp.simplify(S0_g[k, k]) for k in range(3)]

print("mean of the diagonal   m =", m_g)
for k, s_k in enumerate(sigma, start=1):
    print(f"  sigma_{k} = J[{k - 1},{k - 1}] - m =", s_k)
print("\ndiagonal of DF = (m + sigma_1, m + sigma_2, m + sigma_3) =")
print("  ", [sp.simplify(m_g + s_k) for s_k in sigma])
print(
    "\nsigma_1 + sigma_2 + sigma_3 =",
    sp.simplify(sum(sigma)),
    " -> 2 free numbers, not 3",
)
print("div DF =", sp.expand(Jg.trace()), " -> touches the diagonal only through m")

assert sp.simplify(Jg - (iso_g + S0_g + A_g)).is_zero_matrix  # the split is exact
assert sp.simplify(sum(sigma)) == 0  # only two sigmas are free
assert sp.simplify(S0_g.trace()) == 0  # S0 is trace-free
assert sp.simplify(S0_g - S0_g.T).is_zero_matrix  # ... and symmetric
assert sp.simplify(A_g + A_g.T).is_zero_matrix  # A is antisymmetric
assert sp.simplify(Jg.trace() - 3 * m_g) == 0  # div fixes m, and nothing more
S0_g


# %%
# A field with all three parts nonzero: dilation + strain + swirl, plus nonlinearity.
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
    (np.array([[1.0, 0.0], [0.0, -1.0]]), "strain S₀  J = diag(1,−1)", ORANGE),
    (np.array([[0.0, 1.0], [1.0, 0.0]]), "strain S₀  J = [[0,1],[1,0]]", YELLOW),
    (np.array([[0.0, -1.0], [1.0, 0.0]]), "rotation   J = [[0,−1],[1,0]]", GREEN),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5.4))
fig.suptitle(
    "The three atoms of DF:  isotropic scaling (div ≠ 0),  two strains — the same one 45° apart"
    "  (div = curl = 0),  rotation (curl ≠ 0)"
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
# ## 2. Gradient: the operator that *makes* fields with no rotation part
#
# There is a third classical operator, and it sits differently from the other two. Divergence and
# curl take a vector field apart — they are projections of $DF$. **Gradient goes the other way**: it
# takes a scalar function $f:\mathbb{R}^n\to\mathbb{R}$ and *builds* a vector field. Its place in
# this story is that the fields it builds have a very particular $DF$.
#
# ### First: $\nabla f$ is not simply "the derivative of $f$"
#
# The derivative of a scalar function at $p$ is the linear map
#
# $$Df(p):\mathbb{R}^n\to\mathbb{R},\qquad Df(p)\,v \;=\; \left.\frac{d}{dt}\right\rvert_0 f(p+tv),$$
#
# which eats a direction and returns a rate. That is a **row**, not a column — a linear functional,
# not a vector. To *point* somewhere you need to convert "linear functional" into "vector", and the
# thing that does that is an inner product. The gradient is **defined** by
#
# $$\big\langle \nabla f(p),\, v \big\rangle \;=\; Df(p)\,v \quad\text{for every } v .$$
#
# With the ordinary dot product this unpacks to the familiar $\nabla f = (\partial_1 f,\dots,\partial_n f)$
# — which is why the definition is easy to miss. But change the inner product to
# $\langle u,v\rangle_M = u^{\mathsf T}Mv$ (any symmetric positive-definite $M$) and the *same*
# function has a *different* gradient,
#
# $$\nabla_{\!M} f \;=\; M^{-1}\nabla f .$$
#
# Same $f$, same derivative, different arrow. **Gradient is the first place in vector calculus where a
# metric is silently used** — a theme that returns in §3 for curl.


# %%
def grad(f, p, h=1e-6):
    """∇f: the vector of partial derivatives (the gradient for the ordinary dot product)."""
    p = np.asarray(p, float)
    g = np.zeros(p.size)
    for i in range(p.size):
        e = np.zeros(p.size)
        e[i] = h
        g[i] = (f(p + e) - f(p - e)) / (2 * h)
    return g


def hessian(f, p, h=1e-4):
    """D(∇f), the matrix of second partials, by centered differences of the gradient field.

    Accurate (O(h²)) but symmetric *by construction*: the centered stencil gives H[i, j] and H[j, i]
    as the same expression, so this function cannot be used to test equality of mixed partials. Use
    ``hessian_onesided`` for that.
    """
    return jacobian(lambda q: grad(f, q, h), p, h)


def hessian_onesided(f, p, h=1e-4):
    """The same Hessian from a one-sided outer difference — a stencil that is NOT i↔j symmetric.

    Only first-order accurate, but its antisymmetric part is then a genuine measurement rather than
    an identity of the formula, so watching ‖skew‖ → 0 as h → 0 actually tests Clairaut's theorem.
    """
    p = np.asarray(p, float)
    g0 = grad(f, p, h)
    H = np.zeros((p.size, p.size))
    for j in range(p.size):
        e = np.zeros(p.size)
        e[j] = h
        H[:, j] = (grad(f, p + e, h) - g0) / h
    return H


def grad_metric(f, p, M, h=1e-6):
    """∇_M f, the gradient with respect to ⟨u,v⟩_M = uᵀMv — i.e. M⁻¹∇f."""
    return np.linalg.solve(M, grad(f, p, h))


# a scalar function and a non-standard inner product
def fscalar(q):
    x, y = q
    return x**2 * y - 0.5 * y**3 + np.sin(x * y)


M = np.array([[2.0, 0.6], [0.6, 1.0]])  # symmetric positive definite
q = np.array([0.7, -0.4])
v = np.array([0.35, 1.1])

gE = grad(fscalar, q)
gM = grad_metric(fscalar, q, M)
Dv = (fscalar(q + 1e-6 * v) - fscalar(q - 1e-6 * v)) / 2e-6  # directional derivative

print("directional derivative D_v f          :", f"{Dv:+.8f}")
print("⟨∇f, v⟩   with the dot product        :", f"{gE @ v:+.8f}")
print("⟨∇_M f, v⟩_M with the M inner product :", f"{gM @ M @ v:+.8f}")
print("\nboth reproduce D_v f :", np.allclose([gE @ v, gM @ M @ v], Dv, atol=1e-6))
print("but they are different vectors:")
print("   ∇f   =", np.round(gE, 6))
print(
    "   ∇_M f =",
    np.round(gM, 6),
    "  ← same function, different metric, different arrow",
)

# %% [markdown]
# ### Then: a gradient field has $A = 0$, everywhere
#
# Now feed $F = \nabla f$ into §1. Its derivative is the **Hessian**,
#
# $$D(\nabla f) \;=\; \operatorname{Hess} f \;=\; \left[\partial_i\partial_j f\right],$$
#
# and by equality of mixed partials ($\partial_i\partial_j f = \partial_j\partial_i f$, Clairaut's
# theorem) the Hessian is **symmetric**. Read that through the three-way split of §1:
#
# | | for a general $F$ | for $F = \nabla f$ |
# |---|---|---|
# | dilation | $\tfrac{1}{n}(\operatorname{div}F)I$ | $\tfrac{1}{n}(\Delta f)\,I$ |
# | strain $S_0$ | trace-free symmetric | trace-free part of $\operatorname{Hess}f$ |
# | rotation $A$ | $\tfrac12$ the curl | $\;\mathbf{0}$ — always |
#
# Three consequences fall out at once, and none of them is a computation:
#
# 1. **$\operatorname{curl}\nabla f = 0$.** Not an identity you grind out in coordinates — the spin
#    part of a Hessian is zero because the Hessian is symmetric. Symmetry of second partials *is*
#    curl-grad-zero.
# 2. **$\operatorname{div}\nabla f = \operatorname{tr}\operatorname{Hess}f = \Delta f$**, the
#    **Laplacian**. The Laplacian is the trace of the Hessian, exactly as the divergence is the trace
#    of $DF$ — same projection, one step further along.
# 3. **A gradient field never spirals.** A symmetric matrix has real eigenvalues and orthogonal
#    eigenvectors, so near a critical point the flow $\dot x = -\nabla f$ can only stretch or
#    compress along $n$ perpendicular axes. Complex eigenvalues — the source of rotation — require a
#    nonzero antisymmetric part, and a gradient field has none.
#
# The converse of (1) — curl-free $\Rightarrow$ locally a gradient — is the Poincaré lemma, and it
# needs an assumption about the *shape of the domain*. §6 and §8 come back to it.

# %%
# (1) and (2), proven for a GENERIC f with SymPy — true for every smooth scalar function
xs, ys, zs = sp.symbols("x y z", real=True)
f_gen = sp.Function("f")(xs, ys, zs)
Hess = sp.Matrix(3, 3, lambda i, j: sp.diff(f_gen, (xs, ys, zs)[i], (xs, ys, zs)[j]))
print("Hess f symmetric for every f :", sp.simplify(Hess - Hess.T) == sp.zeros(3, 3))
print("  ⇒ antisymmetric part ≡ 0   ⇒ curl ∇f ≡ 0")
lap = sum(sp.diff(f_gen, c, 2) for c in (xs, ys, zs))
print("tr Hess f == Δf              :", sp.simplify(Hess.trace() - lap) == 0)


# (1)–(3) again, numerically on a concrete function of three variables
def fs3(p):
    x, y, z = p
    return x**2 * y + np.sin(y * z) - 0.3 * x * z**2


p = np.array([0.4, -0.7, 1.1])
H = hessian(fs3, p)
gradfield = lambda w: grad(fs3, w, h=1e-4)

print("\nHessian of a concrete f at p:\n", np.round(H, 5))
print(
    "\ncurl(∇f)                           :", np.round(curl3(gradfield, p, h=1e-4), 6)
)
print("div(∇f) = tr Hess                  :", f"{np.trace(H):+.6f}")
print(
    "Δf by direct second differences    :",
    f"{sum((fs3(p + e) - 2 * fs3(p) + fs3(p - e)) / 1e-8 for e in 1e-4 * np.eye(3)):+.6f}",
)
print("eigenvalues of Hess (all real)     :", np.round(np.linalg.eigvalsh(H), 5))
print("  ⇒ the gradient flow has no rotation anywhere: A ≡ 0")

# A real test of Clairaut: with a stencil that does not build the symmetry in, ‖skew‖ → 0 like O(h).
print("\nantisymmetric part of a one-sided Hessian (not symmetric by construction):")
for hh in [1e-2, 1e-3, 1e-4]:
    print(
        f"   h = {hh:7.0e}:  ‖skew(Hess)‖ = {np.linalg.norm(skew(hessian_onesided(fs3, p, hh))):.3e}"
    )
print("   → 0 like O(h): the symmetry belongs to f, not to the difference formula.")

# %% [markdown]
# ### And the special case that ties §1 to the rest of the notebook
#
# Set the *first* summand to zero as well. A gradient field with $\Delta f = 0$ — an **harmonic**
# function — has a Hessian that is symmetric *and* trace-free: it is **pure strain**, the one piece
# of $DF$ that neither divergence nor curl can see.
#
# $$\Delta f = 0 \quad\Longleftrightarrow\quad
# \operatorname{div}\nabla f = 0 \ \text{ and }\ \operatorname{curl}\nabla f = 0
# \quad\Longleftrightarrow\quad \operatorname{Hess}f = S_0 .$$
#
# This is where the notebook's running examples come from. $f = xy$ is harmonic, so
# $\nabla f = (y,\,x)$ has zero divergence and zero curl while $DF\neq 0$ — the field §8 uses to show
# what the two operators miss. Harmonic functions are precisely the scalar potentials whose fields
# are invisible to div and curl, which is also why they are exactly the ambiguity in the Helmholtz
# decomposition.

# %%
harmonics = [
    (lambda q: q[0] * q[1], "f = xy", "∇f = (y, x)"),
    (lambda q: 0.5 * (q[0] ** 2 - q[1] ** 2), "f = (x²−y²)/2", "∇f = (x, −y)"),
    (
        lambda q: np.exp(q[0]) * np.cos(q[1]),
        "f = eˣ cos y",
        "∇f = (eˣcos y, −eˣ sin y)",
    ),
]
q = np.array([0.7, -0.3])
print("harmonic f  ⇒  ∇f is pure strain: div = curl = 0 but DF ≠ 0")
for fh, name, gname in harmonics:
    Hh = hessian(fh, q)
    gf = lambda w: grad(fh, w, h=1e-4)
    print(
        f"  {name:16s} {gname:26s} Δf = {np.trace(Hh):+.2e}"
        f"   curl ∇f = {curl2(gf, q, h=1e-4):+.2e}   ‖Hess‖ = {np.linalg.norm(Hh):.4f}  ← nonzero"
    )

# a non-harmonic f, for contrast: the dilation part survives
Hd = hessian(lambda w: 0.5 * (w[0] ** 2 + w[1] ** 2), q)
print(
    f"\n  f = (x²+y²)/2    ∇f = (x, y)                Δf = {np.trace(Hd):+.4f}   ← a source: div ≠ 0"
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.suptitle(
    "Gradient: perpendicular to the level sets — but only with respect to the inner product used to build it",
    color="white",
    fontsize=11,
)
gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 240), np.linspace(-1.6, 1.6, 240))
qx, qy = np.meshgrid(np.linspace(-1.35, 1.35, 8), np.linspace(-1.35, 1.35, 8))


def unit(U, V):
    """Direction only: here the angle carries the message, not the arrow length."""
    m = np.hypot(U, V)
    d = np.where(m == 0, 1.0, m)
    return U / d, V / d, m


# (a) an ordinary f: ∇f ⟂ the level sets, and the field has no circulation
Fa = lambda X, Y: X**2 * Y - 0.5 * Y**3
axes[0].contour(gx, gy, Fa(gx, gy), levels=15, colors="#7d7da0", linewidths=1.0)
U, V = 2 * qx * qy, qx**2 - 1.5 * qy**2
u, v, mag = unit(U, V)
axes[0].quiver(qx, qy, u, v, mag, cmap="viridis", alpha=0.95, scale=13)
axes[0].set_title(
    "(a)  f = x²y − y³/2,  arrows = ∇f\nperpendicular to every contour;  curl ∇f ≡ 0",
    color="#ccc",
    fontsize=9,
)

# (b) a harmonic f: the gradient field is pure strain — invisible to div and curl
axes[1].contour(gx, gy, gx * gy, levels=17, colors="#7d7da0", linewidths=1.0)
u, v, mag = unit(qy.copy(), qx.copy())
axes[1].quiver(qx, qy, u, v, mag, cmap="plasma", alpha=0.95, scale=13)
axes[1].set_title(
    "(b)  f = xy  is harmonic,  ∇f = (y, x)\nΔf = 0 ⇒ div = 0 AND curl = 0,  yet DF ≠ 0",
    color="#ccc",
    fontsize=9,
)

# (c) the same f as (a), but with the gradient taken in a skewed inner product
axes[2].contour(gx, gy, Fa(gx, gy), levels=15, colors="#7d7da0", linewidths=1.0)
Minv = np.linalg.inv(M)
ue, ve, _ = unit(U, V)
um, vm, _ = unit(Minv[0, 0] * U + Minv[0, 1] * V, Minv[1, 0] * U + Minv[1, 1] * V)
axes[2].quiver(
    qx, qy, ue, ve, color=BLUE, alpha=0.95, scale=13, label="∇f  (dot product)"
)
axes[2].quiver(
    qx, qy, um, vm, color=ORANGE, alpha=0.95, scale=13, label="∇_M f  (M inner product)"
)
axes[2].legend(fontsize=8, loc="upper left", framealpha=0.9)
axes[2].set_title(
    "(c)  same f, two inner products\nthe orange arrows are NOT ⟂ to the contours in this drawing",
    color="#ccc",
    fontsize=9,
)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

print(
    "Panel (c): 'the gradient points uphill, perpendicular to the level sets' is a statement about"
)
print(
    "an inner product, not about f. Change the inner product and the arrow moves; the contours do not."
)

# %% [markdown]
# ## 3. What is invariant: why $\operatorname{tr}$ is cheap and the antisymmetric part is expensive
#
# Change coordinates linearly, $y = Px$. A vector field transforms as $\tilde F(y) = P\,F(P^{-1}y)$,
# so its derivative is **conjugated**:
#
# $$D\tilde F \;=\; P\,(DF)\,P^{-1}.$$
#
# Now the two projections behave very differently:
#
# * $\operatorname{tr}(PJP^{-1}) = \operatorname{tr}J$ for **every** invertible $P$ — the trace does
#   not care which coordinates you use. Divergence therefore survives arbitrary changes of frame; it
#   needs only a notion of *volume*, not of length or angle.
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
# curl, while the divergence is untouched. Divergence is invariant under *all* invertible changes of
# frame; curl transforms correctly only under the **orthogonal** ones — those that preserve the inner
# product. (§2 met the same fact one step earlier: the gradient itself already depends on which inner
# product you use.)
#
# > **On a manifold.** For a vector field $F$ on $M$, $DF$ is only defined once you choose a
# > connection $\nabla$; then $\operatorname{div}F = \operatorname{tr}\nabla F$ and the trace kills the
# > connection-dependence for the Levi-Civita choice. Equivalently, and better, define divergence with
# > no connection at all by $\mathcal{L}_F\mu = (\operatorname{div}_\mu F)\,\mu$ for a volume form $\mu$
# > — §6 shows this is the same thing.

# %% [markdown]
# ## 4. Divergence = rate of change of volume (Liouville)
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
# ## 5. Curl = twice the angular velocity (Cauchy–Stokes)
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
# ## 6. The Spivak view: it is all $d$
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
# two), living in the space $\Lambda^2$ of 2-forms. The second says: **the exterior derivative of
# $\iota_F\mu$ *is* the divergence** — which, by Cartan's magic formula
# $\mathcal{L}_F\mu = d\iota_F\mu + \iota_F\,d\mu = d\iota_F\mu$ (since $d\mu = 0$), is exactly the
# volume statement of §4:
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
# ## 7. Why curl is a vector only when $n = 3$
#
# The antisymmetric part $A$ is an antisymmetric $n\times n$ matrix, and the space of those has
# dimension $\binom{n}{2}=\tfrac{n(n-1)}{2}$ — just count the entries strictly above the diagonal.
# The hat/vee bijection of the toolkit turned such a matrix into a vector; for that to be possible at
# all you need
#
# $$\binom{n}{2} \;=\; n \qquad\Longleftrightarrow\qquad n = 3 .$$
#
# That is the whole story of the "vector curl". In other dimensions the object $dF^\flat$ still exists
# and is still the antisymmetric part of $DF$ — it just is not a vector:
#
# | $n$ | antisymmetric matrices: $\binom{n}{2}$ | what "curl" is | decomposition $n^2 = 1 + (\tfrac{n(n+1)}{2}-1) + \binom{n}{2}$ |
# |---|---|---|---|
# | 2 | 1 | a **scalar** $\partial_xF_y-\partial_yF_x$ | $4 = 1 + 2 + 1$ |
# | 3 | 3 | a **vector** (pseudo-vector) | $9 = 1 + 5 + 3$ |
# | 4 | 6 | a **bivector** — e.g. $F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu$ in electromagnetism | $16 = 1 + 9 + 6$ |
# | $n$ | $\binom{n}{2}$ | a 2-form | — |
#
# The electromagnetic field tensor is literally "the curl of the 4-potential": $F = dA$, and its six
# components are $\mathbf E$ and $\mathbf B$ — three "electric" and three "magnetic" only because
# $\binom{4}{2} = 3 + 3$. The cross product and the vector curl are $n=3$ coincidences; the
# antisymmetric matrix — equivalently the 2-form of §6 — is the invariant object.

# %%
print("n×n matrices:  n² = 1 (dilation) + [n(n+1)/2 − 1] (shear) + n(n−1)/2 (rotation)")
for n in range(2, 8):
    a, b, c = 1, n * (n + 1) // 2 - 1, n * (n - 1) // 2
    assert a + b + c == n * n
    print(
        f"  n = {n}:  {n * n:3d} = {a} + {b:2d} + {c:2d}"
        f"   curl is a {'scalar' if c == 1 else 'vector' if c == n else f'{c}-component 2-form'}"
        f"{'   ← the only n with as many antisymmetric matrices as dimensions' if c == n and n > 2 else ''}"
    )

# in 2D the whole antisymmetric part is one number, and it IS the scalar curl
Jp = jacobian(Fcomp, p2)
print(f"\n2D: the antisymmetric part has a single free entry:\n{np.round(skew(Jp), 6)}")
print(
    f"    its single free entry ×2 = {2 * skew(Jp)[1, 0]:.6f} == scalar curl {curl2(Fcomp, p2):.6f}"
)

# %% [markdown]
# ## 8. What div and curl do *not* see — and what they nevertheless determine
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
# Its potential $xy$ is harmonic, which by §2 is exactly the statement
# $\operatorname{div}=\operatorname{curl}=0$: a gradient field has no spin part at all, and a
# *harmonic* potential kills the dilation part too, leaving nothing but strain.

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
# ## 9. Coda: Jacobians of learned flows
#
# The same decomposition is the working vocabulary for the Jacobian of a learned map, which may be of
# interest if you look at latent-space geometry.
#
# * **Continuous normalizing flows.** For $\dot z = F_\theta(z)$, the instantaneous change-of-variables
#   formula is $\frac{d}{dt}\log p(z(t)) = -\operatorname{div}F_\theta(z(t))$ — literally §4, Liouville
#   run backwards. This is the identity that makes neural ODEs tractable: you need only the *trace* of
#   the Jacobian, one of the three summands, and it can be estimated with a single matrix-vector
#   product by Hutchinson's estimator $\operatorname{tr}J = \mathbb{E}_v[v^{\mathsf T}Jv]$.
# * **The other two summands are what interpretability work is usually after.** The trace-free
#   symmetric part $S_0$ is the local anisotropic stretching — its spectrum is the local condition
#   number, the thing that shows up as the singular-value spectrum of the Jacobian and controls how a
#   metric is distorted between latent and data space. The antisymmetric part $A$ is the local frame
#   rotation — the part that moves features without changing any density or any length.
# * A residual block $x \mapsto x + \varepsilon f(x)$ has Jacobian $I + \varepsilon Df$, i.e. exactly
#   the $t\to 0$ flow of §5: dilation controls the log-volume, the strain part controls conditioning,
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
# ## 10. Advanced note: why these three pieces, and why *only* div and curl
#
# *Everything above needs only a first course in linear algebra. This closing section names the
# structure in the language of group representations, and uses it to prove something the earlier
# sections could only assert: that divergence and curl are not merely two natural operators built
# from $DF$ — up to scale they are the **only** ones.*
#
# ### The names
#
# Write $\mathfrak{gl}(n)$ for the vector space of all real $n\times n$ matrices (the Lie algebra of
# the invertible group $GL(n)$). The rotation group $SO(n)$ acts on it by conjugation,
# $J\mapsto RJR^{\mathsf T}$. The splitting of §1 is then
#
# $$\mathfrak{gl}(n)\;=\;\underbrace{\mathbb{R}\,I}_{\dim 1}\;\oplus\;
#   \underbrace{\operatorname{Sym}_0(n)}_{\dim\frac{n(n+1)}{2}-1}\;\oplus\;
#   \underbrace{\mathfrak{so}(n)}_{\dim\frac{n(n-1)}{2}},$$
#
# and this is the decomposition into **irreducible representations** of $SO(n)$: each summand is
# carried to itself by every rotation, and none of them contains a smaller subspace with that
# property. That is the precise sense in which the three-way split is canonical rather than a
# convenient choice — it is the *finest* decomposition compatible with rotational symmetry, so any
# rotationally sensible way of taking $DF$ apart must be built from these three blocks and no others.
#
# In $\mathbb{R}^3$ the summands have dimensions $1,5,3$ and are the familiar spin
# $0\oplus2\oplus1$ of a rank-2 Cartesian tensor: $\mathbf 3\otimes\mathbf 3 = \mathbf 1\oplus\mathbf
# 5\oplus\mathbf 3$. The antisymmetric summand $\mathfrak{so}(n)$ is the same thing as
# $\Lambda^2(\mathbb{R}^n)$, the 2-forms of §6 — which is why the exterior derivative $dF^\flat$ and
# the spin part of $DF$ are one object, and why "curl is a vector" needs $\dim\mathfrak{so}(3)=3$.
#
# ### The uniqueness statement
#
# Complete reducibility plus **Schur's lemma** turn the decomposition into a counting theorem: the
# space of $SO(n)$-equivariant linear maps out of $\mathfrak{gl}(n)$ into a given irreducible
# representation $W$ has dimension equal to the **multiplicity of $W$ in $\mathfrak{gl}(n)$**. So:
#
# * **Scalars.** The trivial representation appears in $\mathfrak{gl}(n)$ with multiplicity **1** for
#   $n\ge3$ — only the $\mathbb{R}I$ line. Hence, up to a constant, $\operatorname{tr}$ is the *only*
#   rotation-invariant linear functional of $DF$: **divergence is the unique first-order rotationally
#   invariant scalar you can build from a vector field.**
# * **Vectors.** The defining representation $\mathbb{R}^3$ appears in $\mathfrak{gl}(3)$ with
#   multiplicity **1** — only $\mathfrak{so}(3)$. Hence **curl is the unique first-order equivariant
#   vector**, up to scale. (The factor of 2 in $\operatorname{curl}=2\,\mathrm{vee}(A)$ is exactly the
#   "up to scale" freedom; §5 shows why $\tfrac12$ is the physically meaningful normalization.)
# * **$n=2$ is the exception.** $SO(2)$ is abelian and fixes $\mathfrak{so}(2)$ pointwise, so the
#   trivial representation appears with multiplicity **2**. There are two invariant scalars in the
#   plane — divergence *and* the scalar curl. (Only $SO(2)$-invariant, not $O(2)$: a reflection flips
#   the sign of the scalar curl, which is what "pseudoscalar" means.)
#
# All three multiplicities can be measured. Averaging the conjugation action over the group,
# $P(J)=\big\langle RJR^{\mathsf T}\big\rangle_{R\in SO(n)}$, gives a projection onto the invariant
# subspace, so **the rank of $P$ is the multiplicity of the trivial representation**. We build $P$ by
# Monte-Carlo averaging over random rotations and read off its rank.


# %%
def random_rotation(rng, n):
    """A Haar-uniform element of SO(n), via a QR decomposition of a Gaussian matrix."""
    Q, R = np.linalg.qr(rng.normal(size=(n, n)))
    Q = Q * np.sign(np.diag(R))  # fix the QR sign ambiguity → Haar measure on O(n)
    if np.linalg.det(Q) < 0:  # then restrict to SO(n)
        Q[:, 0] *= -1
    return Q


def conjugation_matrices(rng, n, m):
    """m samples of the linear map J ↦ R J Rᵀ, each as an n²×n² matrix."""
    Rs = np.array([random_rotation(rng, n) for _ in range(m)])
    # (R J Rᵀ)_{ik} = R_{ia} J_{ab} R_{kb}   ⇒   operator[(i,k), (a,b)] = R_{ia} R_{kb}
    return np.einsum("mia,mkb->mikab", Rs, Rs).reshape(m, n * n, n * n), Rs


rng = np.random.default_rng(0)
print(
    "multiplicity of the trivial representation in gl(n)  =  rank of the averaging operator P"
)
print("(= how many independent rotation-invariant scalars can be built from DF)\n")
for n in (2, 3, 4):
    ops, _ = conjugation_matrices(rng, n, 40_000)
    P = ops.mean(axis=0)
    sv = np.linalg.svd(P, compute_uv=False)
    rank = int(np.sum(sv > 0.1))  # Monte-Carlo noise floor is ~1/√m ≈ 0.005
    print(
        f"  n = {n}:  rank P = {rank}   singular values {np.round(sv[:4], 4)} …"
        f"   {'div and the scalar curl' if rank == 2 else 'div alone'}"
    )

# for n = 3 the image really is the multiples of the identity
ops, _ = conjugation_matrices(np.random.default_rng(1), 3, 40_000)
P3 = ops.mean(axis=0)
U, sv, _ = np.linalg.svd(P3)
image = U[:, 0].reshape(3, 3)
image = image / image[0, 0]
print("\nfor n = 3 the one invariant direction, normalized:\n", np.round(image, 4))
print(
    "  ⇒ the only rotation-invariant linear functional of DF is J ↦ c·tr J:  divergence is unique."
)

# %% [markdown]
# Now the vector statement. A first-order equivariant vector built from $DF$ is a linear map
# $L:\mathfrak{gl}(3)\to\mathbb{R}^3$ satisfying $L(RJR^{\mathsf T}) = R\,L(J)$ for every rotation.
# That is a linear condition on the 27 entries of $L$; the dimension of its solution space is the
# multiplicity we want. We solve it numerically and check that the answer is one-dimensional and
# spanned by $\mathrm{vee}\circ\operatorname{skew}$.

# %%
n = 3
rng = np.random.default_rng(2)
ops, Rs = conjugation_matrices(rng, n, 60)

# unknowns: the 3×9 matrix L.  Constraint for each R:  L·P_R − R·L = 0   (3×9 = 27 equations)
rows = []
basis = np.eye(3 * 9).reshape(3 * 9, 3, 9)
for P_R, R in zip(ops, Rs):
    for Lb in basis:
        rows.append((Lb @ P_R - R @ Lb).reshape(-1))
Aeq = np.array(rows).reshape(len(ops), 27, 27).transpose(0, 2, 1).reshape(-1, 27)

sv = np.linalg.svd(Aeq, compute_uv=False)
nullity = int(np.sum(sv < 1e-8 * sv[0]))
print(f"equivariant linear maps gl(3) → ℝ³:  solution space has dimension {nullity}")

Lsol = np.linalg.svd(Aeq)[2][-1].reshape(3, 9)
Lcurl = np.array(
    [vee(skew(np.eye(9)[k].reshape(3, 3))) for k in range(9)]
).T  # vee∘skew as a 3×9
scale = float(Lsol.ravel() @ Lcurl.ravel()) / float(
    Lcurl.ravel() @ Lcurl.ravel()
)  # least squares
print("  the recovered map, rescaled:\n", np.round(Lsol / scale, 6))
print("  vee∘skew itself:\n", np.round(Lcurl, 6))
print(
    "the solution is proportional to vee∘skew :",
    np.allclose(Lsol, scale * Lcurl, atol=1e-8),
)
print(
    "  ⇒ up to a constant, curl is the ONLY first-order equivariant vector built from DF."
)

# and the three summands are each carried to themselves by rotations (that is what "invariant" means)
J = jacobian(F3, np.array([0.4, -0.7, 1.1]))
R = random_rotation(np.random.default_rng(3), 3)
print("\nrotating the frame maps each summand into itself:")
print(
    "   iso (R J Rᵀ) == R iso(J) Rᵀ :", np.allclose(iso(R @ J @ R.T), R @ iso(J) @ R.T)
)
print(
    "   dev (R J Rᵀ) == R dev(J) Rᵀ :", np.allclose(dev(R @ J @ R.T), R @ dev(J) @ R.T)
)
print(
    "   skew(R J Rᵀ) == R skew(J) Rᵀ:",
    np.allclose(skew(R @ J @ R.T), R @ skew(J) @ R.T),
)

# %% [markdown]
# So the picture the notebook opened with is not a convenient way of slicing $DF$ — it is the only
# rotationally coherent way, and divergence and curl exhaust the scalars and vectors it can produce.
# Everything else about $DF$ lives in the 5-dimensional strain block, which by the same counting
# argument cannot be reduced to a scalar or a vector at all. That is the structural reason §8's
# "invisible" fields are invisible: there is no operator of div-or-curl type left to see them with.
#
# **Further reading.** H. Weyl, *The Classical Groups*, Princeton, 1939 (the decomposition of tensor
# representations of the orthogonal group); W. Fulton and J. Harris, *Representation Theory: A First
# Course*, Springer, 1991, §§1–3 for Schur's lemma and multiplicity counting, Ch. 19 for
# $\mathfrak{so}(n)$; and for the physics dictionary between $\mathbf 3\otimes\mathbf 3 = \mathbf
# 1\oplus\mathbf 5\oplus\mathbf 3$ and multipole moments, J. D. Jackson, *Classical Electrodynamics*,
# 3rd ed., Wiley, 1998, §4.1.

# %% [markdown]
# ## Summary
#
# | question | answer | where it lives in $DF$ |
# |---|---|---|
# | What is $\operatorname{grad}f$? | the vector representing $Df$ under an inner product | not a part of $DF$ — it *builds* fields with $A=0$ |
# | What is $\operatorname{div}F$? | $\operatorname{tr}DF$ | the isotropic part $\tfrac{\operatorname{tr}J}{n}I$ |
# | What is $\operatorname{curl}F$? | $2\,\mathrm{vee}(\operatorname{skew}DF)$, i.e. $dF^\flat$ | the antisymmetric part $A$ |
# | What do they miss? | the rate of strain $S_0$ | the trace-free symmetric part — 5 of 9 components in $\mathbb{R}^3$ |
# | Why is $\operatorname{div}$ a trace? | $\left.\tfrac{d}{dt}\right\rvert_0\det D\varphi_t=\operatorname{tr}DF$ | volume rate (Liouville) |
# | Why the $\tfrac12$ in curl? | $A$ *is* the angular velocity; curl is $2A$ | mean spin of material line elements |
# | What structure does each need? | $\operatorname{grad}$, $\operatorname{curl}$: an inner product ($+$ an orientation for a vector curl). $\operatorname{div}$: only a volume | trace survives any change of frame; transpose does not |
# | Why is curl a vector only in 3D? | antisymmetric $n\times n$ matrices number $\binom{n}{2}$, and $\binom{n}{2}=n$ only at $n=3$ | the hat/vee bijection |
# | Why is $\operatorname{curl}\operatorname{grad}=0$? | $D(\nabla f)=\operatorname{Hess}f$ is symmetric | a gradient field has no antisymmetric part |
# | Why is $\Delta=\operatorname{div}\operatorname{grad}$? | $\operatorname{tr}\operatorname{Hess}f$ | the same trace, one step further along |
# | What is the invariant statement? | $df$, $dF^\flat=2A$, and $d\,\iota_F\mu=(\operatorname{div}F)\mu$ | one operator $d$, three degrees |
#
# The single sentence: **$DF(p)$ is a square matrix acting on the tangent space; every square matrix
# splits into dilation $+$ strain $+$ spin; divergence reads the first part and curl the third, while
# gradient is the operator that manufactures the fields whose third part is zero.** That is a complete
# account of $DF$ only when the strain vanishes, and a complete account of $F$ only after adding
# global (topological) information — and §10 shows there is no fourth operator waiting to be found.
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
#   dilation, strain and rigid rotation (the Cauchy–Stokes decomposition of §5).
# - H. Helmholtz, "Über Integrale der hydrodynamischen Gleichungen, welche den Wirbelbewegungen
#   entsprechen", *J. reine angew. Math.* **55** (1858), 25–55; English translation by P. G. Tait,
#   *Phil. Mag.* **33** (1867), 485–512 — the decomposition and the vortex theorems.
# - J. Liouville, "Sur la théorie de la variation des constantes arbitraires", *J. Math. Pures Appl.*
#   **3** (1838), 342–349 — the formula of §4.
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
# - `LieGroups/SO3_Lie_Theory.py` — the hat map from $\mathbb{R}^3$ to antisymmetric matrices, used here to turn $A$
#   into $\tfrac12\operatorname{curl}F$.
