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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric Linear Algebra 5 — Coordinates, Matrices, and the Discovery of ℂ
#
# For four notebooks we held coordinates at arm's length: vectors were arrows, operators were
# geometric actions, and the determinant and trace were defined by what they *do* to area and volume.
# Now we introduce the missing machinery — a **basis**, the **coordinates** of a vector, and the
# **matrix** of an operator — and discover it adds *no new content*: matrices are just a bookkeeping
# language for the operators we already understand. Determinant and trace reappear as the familiar
# formulas $ad-bc$ and $a+d$, and we verify they are **basis-independent**, which is exactly why the
# coordinate-free definitions were legitimate.
#
# Then the payoff. We single out the operators that are **a scaling combined with a pure rotation** —
# a *scale-and-twist* — and find that they form a closed, commutative algebra. That algebra **is the
# complex numbers**: a complex number $z = x + iy$ literally *is* the operator $xI + yJ$, where $J$ is
# the quarter-turn from notebook 2 with $J^2 = -I$. Multiplying complex numbers is composing
# scale-and-twists; $|z|^2$ is its area-scaling determinant; $e^{i\theta}$ is rotation by $\theta$.
#
# This is the mirror image of the `LieGroups/` thread, which discovers **SO(2) ≅ U(1)** from the
# rotation *group*; here ℂ falls out of the linear *operators* of elementary plane geometry. k3d as
# always.

# %%
import numpy as np
from scipy.linalg import expm
import k3d

BLUE = 0x4FC3F7
ORANGE = 0xFFB74D
GREEN = 0x81C784
RED = 0xEF5350
PURPLE = 0xCE93D8
GREY = 0x777777
FAINT = 0x3A3A4E

BG = 0x0F0F0F
GRIDC = 0x333333
LABELC = 0xCCCCCC


# %% [markdown]
# ### Toolkit (same helpers as the rest of the series)


# %%
def _to3(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[-1] == 3:
        return pts
    zeros = np.zeros(pts.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate([pts, zeros], axis=-1)


def new_plot(lim=3.0, top_down=True, axes=True, grid=True):
    plot = k3d.plot(
        background_color=BG, grid_color=GRIDC, label_color=LABELC,
        grid_visible=grid, camera_auto_fit=not top_down, menu_visibility=False,
    )
    if top_down:
        plot.camera = [0, 0, 2.6 * lim, 0, 0, 0, 0, 1, 0]
    if axes:
        add_line(plot, np.array([[-lim, 0], [lim, 0]]), GREY, width=0.012)
        add_line(plot, np.array([[0, -lim], [0, lim]]), GREY, width=0.012)
    return plot


def add_line(plot, pts, color, width=0.02, alpha=1.0):
    plot += k3d.line(_to3(pts), color=color, width=width, shader="thick", opacity=alpha)


def add_vector(plot, tail, head, color, label=None, label_size=0.7):
    tail3 = _to3(np.atleast_2d(tail))
    head3 = _to3(np.atleast_2d(head))
    plot += k3d.vectors(origins=tail3, vectors=(head3 - tail3), color=color, head_size=1.5, line_width=0.03)
    if label:
        pos = (tail3[0] + 0.55 * (head3[0] - tail3[0])).tolist()
        plot += k3d.text(label, position=pos, color=color, size=label_size, label_box=False)


def add_points(plot, pts, color, size=0.18):
    plot += k3d.points(_to3(np.atleast_2d(pts)), color=color, point_size=size, shader="3d")


FLAG = np.array([[0.0, 0.0], [0.0, 1.6], [1.0, 1.3], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0]])


def show_operator(matfunc, color, lim=2.0):
    """Overlay a unit grid (faint) with its image under a 2x2 matrix action (bold) + the flag."""
    p = new_plot(lim=2.6 * lim, axes=True)
    ticks = np.linspace(-lim, lim, 9)
    s = np.linspace(-lim, lim, 30)
    for tk in ticks:
        for ln in (np.stack([s, np.full_like(s, tk)], 1), np.stack([np.full_like(s, tk), s], 1)):
            add_line(p, ln, FAINT, width=0.01)
            add_line(p, ln @ matfunc.T, color, width=0.018)
    add_line(p, FLAG, FAINT, width=0.02)
    add_line(p, FLAG @ matfunc.T, color, width=0.04)
    add_points(p, [0, 0], GREEN, size=0.16)
    return p


# %% [markdown]
# ## 1. A basis turns arrows into coordinates
#
# Pick two arrows $b_1, b_2$ that are not parallel — a **basis**. Then *every* arrow $v$ is a unique
# linear combination
#
# $$v = x\,b_1 + y\,b_2,$$
#
# and the pair $(x, y)$ is **the coordinates of $v$ in that basis**. Coordinates are not intrinsic to
# the arrow — they are a *report relative to a chosen frame*. Change the basis and the very same arrow
# gets different coordinates. We illustrate with the standard basis and a skewed one.


# %%
def coords_in(basis, v):
    """Coordinates (x,y) of arrow v in the given basis (columns b1, b2): solve B x = v."""
    return np.linalg.solve(np.asarray(basis, float).T, np.asarray(v, float))


std = np.array([[1.0, 0.0], [0.0, 1.0]])  # standard basis e1, e2 (rows)
skew = np.array([[1.0, 0.0], [0.6, 1.0]])  # a sheared basis b1, b2 (rows)

v = np.array([1.8, 1.2])
cs, ck = coords_in(std, v), coords_in(skew, v)
print("arrow v (a fixed geometric object):", v)
print("  coordinates in standard basis :", np.round(cs, 3))
print("  coordinates in skewed basis   :", np.round(ck, 3))
print("  reconstruct from skew coords  :", np.round(ck @ skew, 3), "→ same arrow:", np.allclose(ck @ skew, v))

p = new_plot(lim=3.0)
# standard frame (grey) and skew frame (purple); v as a combination of the skew basis.
add_vector(p, [0, 0], std[0], GREY, "e1")
add_vector(p, [0, 0], std[1], GREY, "e2")
add_vector(p, [0, 0], skew[0], PURPLE, "b1")
add_vector(p, [0, 0], skew[1], PURPLE, "b2")
add_vector(p, [0, 0], v, BLUE, "v")
add_line(p, np.array([ck[0] * skew[0], v]), ORANGE, width=0.015)  # the y·b2 leg
add_line(p, np.array([[0, 0], ck[0] * skew[0]]), ORANGE, width=0.015)  # the x·b1 leg
p.display()
print("v = x·b1 + y·b2 with (x,y) =", np.round(ck, 3), " (the orange path walks the two legs)")


# %% [markdown]
# ## 2. The matrix of an operator = where it sends the basis
#
# A linear operator is pinned down by what it does to the basis (notebook 1). Write each image
# $T(b_j)$ in coordinates and **stack those coordinate-columns** — that array is the **matrix of $T$**.
# Then "apply $T$" becomes "matrix times coordinate-vector," which is nothing but *take that linear
# combination of the columns*:
#
# $$T(v) = T(x b_1 + y b_2) = x\,T(b_1) + y\,T(b_2) \;\;\longleftrightarrow\;\; \begin{pmatrix}|&|\\ T b_1 & T b_2\\ |&|\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}.$$
#
# So the coordinate-free operators of notebook 2 acquire their familiar matrices in the standard basis.
# We rebuild a few and confirm the matrix action matches the geometric action exactly.


# %%
def J(v):  # the quarter-turn from notebook 2
    v = np.asarray(v, float)
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


# Build each operator's standard matrix by columns = images of e1, e2.
def matrix_of(op):
    return np.array([op(np.array([1.0, 0.0])), op(np.array([0.0, 1.0]))]).T


theta = np.deg2rad(35)
a_dir = np.array([1.0, 0.5])
rotate = lambda v: np.cos(theta) * np.asarray(v, float) + np.sin(theta) * J(v)
project = lambda v: (np.dot(v, a_dir) / np.dot(a_dir, a_dir)) * a_dir
shear = lambda v: np.asarray(v, float) + 0.8 * np.asarray(v, float)[..., 1] * np.array([1.0, 0.0])

for name, op, expected in [
    ("rotation 35°", rotate, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])),
    ("shear k=0.8", shear, np.array([[1.0, 0.8], [0.0, 1.0]])),
]:
    M = matrix_of(op)
    print(f"{name:<14} matrix =\n{np.round(M, 3)}   matches closed form: {np.allclose(M, expected)}")

# Matrix action == geometric action, on random vectors.
Mrot = matrix_of(rotate)
test = np.random.default_rng(5).normal(size=(4, 2))
print("matrix·v reproduces the geometric rotation:", np.allclose(test @ Mrot.T, rotate(test)))

show_operator(matrix_of(shear), ORANGE).display()
print("the shear operator, now as a matrix acting on coordinates — same picture as notebook 2")


# %% [markdown]
# ## 3. Determinant and trace, recovered — and basis-independent
#
# With a matrix $\left(\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\right)$ in hand, the coordinate-free
# definitions collapse to the formulas everyone memorizes:
#
# $$\det = ad - bc \quad(\text{the wedge of the columns}), \qquad \operatorname{tr} = a + d.$$
#
# The crucial check: the *value* must not depend on the basis we happened to choose, or the
# notebook-4 definitions (which never mentioned a basis) would have been nonsense. Re-expressing an
# operator in another basis conjugates its matrix, $A \mapsto P^{-1} A P$ — and both $\det$ and
# $\operatorname{tr}$ are **invariant** under that. We verify.


# %%
A = np.array([[1.3, -0.7], [0.4, 1.1]])
print("det  ad−bc      :", round(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0], 6), " = np.linalg.det:", round(np.linalg.det(A), 6))
print("trace a+d       :", round(A[0, 0] + A[1, 1], 6), " = np.trace:", round(np.trace(A), 6))

rng = np.random.default_rng(6)
inv_ok = True
for _ in range(1000):
    P = rng.normal(size=(2, 2))
    if abs(np.linalg.det(P)) < 1e-6:
        continue
    B = np.linalg.inv(P) @ A @ P  # same operator, different basis
    inv_ok &= np.allclose(np.linalg.det(B), np.linalg.det(A)) and np.allclose(np.trace(B), np.trace(A))
print("det and trace are basis-independent (under A ↦ P⁻¹AP):", inv_ok)


# %% [markdown]
# ## 4. The discovery of ℂ: scale-and-twist operators
#
# Now the destination. Among all linear operators, single out the **scale-and-twist**: scale every
# arrow by some $r\ge 0$ *and* rotate it by some angle $\phi$. From notebook 2, rotation is
# $\cos\phi\,I + \sin\phi\,J$, so a scale-and-twist is
#
# $$r(\cos\phi\,I + \sin\phi\,J) = x\,I + y\,J, \qquad x = r\cos\phi,\; y = r\sin\phi.$$
#
# So the scale-and-twists are exactly the **real combinations $xI + yJ$** of the identity and the
# quarter-turn. As a matrix in the standard basis (using $I=\left(\begin{smallmatrix}1&0\\0&1\end{smallmatrix}\right)$,
# $J=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$):
#
# $$xI + yJ = \begin{pmatrix} x & -y \\ y & x\end{pmatrix}.$$
#
# Watch this set of operators. It is a **2-dimensional plane** (spanned by $I$ and $J$), and — the key
# fact — it is **closed under composition**, because the only product we need is $J^2 = -I$. Let us
# verify closure and that, unlike operators in general, **these commute**.


# %%
def stw(x, y):
    """The scale-and-twist operator xI + yJ as a 2x2 matrix."""
    return np.array([[x, -y], [y, x]], float)


I2, Jm = stw(1, 0), stw(0, 1)
print("J² = −I            :", np.allclose(Jm @ Jm, -I2))

c1, c2 = rng.normal(size=2), rng.normal(size=2)
A1, A2 = stw(*c1), stw(*c2)
prod = A1 @ A2
# The product is again of the form xI+yJ — read off its (x,y) and rebuild.
x_out, y_out = prod[0, 0], prod[1, 0]
print("closed under product:", np.allclose(prod, stw(x_out, y_out)))
print("and they commute     :", np.allclose(A1 @ A2, A2 @ A1))
# The composed (x,y) follow a familiar rule:
print(f"  ({c1[0]:.2f},{c1[1]:.2f}) ∘ ({c2[0]:.2f},{c2[1]:.2f}) = ({x_out:.3f},{y_out:.3f})")
print("  matches (x1x2−y1y2, x1y2+x2y1):",
      np.allclose([x_out, y_out], [c1[0] * c2[0] - c1[1] * c2[1], c1[0] * c2[1] + c2[0] * c1[1]]))


# %% [markdown]
# That composition rule — $(x_1,y_1)\!*\!(x_2,y_2) = (x_1x_2 - y_1y_2,\; x_1y_2 + x_2y_1)$ — is
# **exactly complex multiplication**. The dictionary
#
# $$x\,I + y\,J \;\longleftrightarrow\; x + i\,y$$
#
# is an isomorphism: addition ↔ addition, composition ↔ multiplication, $J \leftrightarrow i$,
# $J^2=-I \leftrightarrow i^2=-1$. **The complex numbers are the algebra of scale-and-twist operators
# of the plane.** We confirm the dictionary respects both operations.


# %%
def to_complex(M):
    return complex(M[0, 0], M[1, 0])  # x + i y


for _ in range(2000):
    z1, z2 = rng.normal(size=2), rng.normal(size=2)
    M1, M2 = stw(*z1), stw(*z2)
    c1c, c2c = to_complex(M1), to_complex(M2)
    assert np.isclose(to_complex(M1 + M2), c1c + c2c)        # addition ↔ addition
    assert np.isclose(to_complex(M1 @ M2), c1c * c2c)        # composition ↔ multiplication
print("matrix algebra {xI+yJ} ≅ ℂ  (addition and composition both match) ✓")


# %% [markdown]
# ### Modulus, argument, determinant, trace — all geometric
#
# Reading $z = xI + yJ$ as the operator "scale by $r$, rotate by $\phi$":
#
# - **modulus** $|z| = r = \sqrt{x^2+y^2}$ is the **length-scaling** factor;
# - **argument** $\arg z = \phi$ is the **rotation angle**;
# - multiplying complex numbers **multiplies moduli and adds arguments** — because composing
#   scale-and-twists multiplies the scales and adds the angles;
# - the **determinant** of $\left(\begin{smallmatrix}x&-y\\y&x\end{smallmatrix}\right)$ is $x^2+y^2 = |z|^2$ — the
#   **area-scaling** factor (notebook 4), and indeed a scale-by-$r$ multiplies area by $r^2$;
# - the **trace** is $2x = 2\,\mathrm{Re}(z) = 2r\cos\phi$.
#
# And the capstone identity, the matrix exponential of the quarter-turn — **Euler's formula** as a
# theorem about operators:
#
# $$e^{\phi J} = \cos\phi\,I + \sin\phi\,J \;\longleftrightarrow\; e^{i\phi} = \cos\phi + i\sin\phi.$$


# %%
z = np.array([1.3, 0.9])  # x, y
M = stw(*z)
r, phi = np.hypot(*z), np.arctan2(z[1], z[0])
print(f"z = {z[0]} + {z[1]}i   modulus r = {r:.4f}   argument φ = {np.rad2deg(phi):.2f}°")
print("det = |z|²        :", np.allclose(np.linalg.det(M), r**2))
print("trace = 2 Re z    :", np.allclose(np.trace(M), 2 * z[0]))

# Moduli multiply, arguments add.
w = np.array([0.5, 1.4])
zw = stw(*z) @ stw(*w)
print("|zw| = |z||w|     :", np.allclose(np.hypot(zw[0, 0], zw[1, 0]), np.hypot(*z) * np.hypot(*w)))
print("arg(zw)=argz+argw :", np.allclose(np.arctan2(zw[1, 0], zw[0, 0]), np.arctan2(z[1], z[0]) + np.arctan2(w[1], w[0])))

# Euler: e^{φJ} is rotation by φ.
phi0 = 0.7
print("e^{φJ} = R(φ)     :", np.allclose(expm(phi0 * Jm),
      np.array([[np.cos(phi0), -np.sin(phi0)], [np.sin(phi0), np.cos(phi0)]])))


# %% [markdown]
# ### Complex multiplication, seen as a scale-and-twist
#
# Multiplying by $z = 1.3 + 0.9i$ scales the plane by $|z|\approx1.58$ and rotates it by
# $\arg z\approx34.7°$. The flag below is carried to its image by exactly that scale-and-twist — a
# single complex multiplication.

# %%
show_operator(stw(*z), GREEN).display()
print(f"multiply-by-z: scale ×{r:.3f}, rotate {np.rad2deg(phi):.1f}° — one complex number, one operator")


# %% [markdown]
# ## Summary — and the arc of the series
#
# - A **basis** assigns each arrow **coordinates**; they are a frame-dependent report, not intrinsic.
# - The **matrix** of an operator stacks the coordinates of the basis images; matrix·vector is just
#   "that linear combination of the columns." Matrices add no new content — they *write down* the
#   operators of notebook 2.
# - **Determinant** $=ad-bc$ (wedge of columns) and **trace** $=a+d$ recover the coordinate-free
#   definitions of notebook 4, and are **basis-independent** (invariant under $A\mapsto P^{-1}AP$) —
#   which is what made those definitions well-posed.
# - The operators that **scale and twist** are precisely $xI+yJ$, closed and commutative under
#   composition because $J^2=-I$. That algebra **is ℂ**: $z=x+iy \leftrightarrow xI+yJ$, with
#   $|z|$ = length scaling, $\arg z$ = rotation, $|z|^2=\det$ = area scaling, $e^{i\phi}=e^{\phi J}$ =
#   rotation. **Complex numbers are the scale-and-twists of the plane.**
#
# **The whole arc.** We began with arrows and two operations (nb 1), defined the operators of geometry
# without coordinates (nb 2), built the dot and wedge to measure length, angle, area, and volume
# (nb 3), distilled volume into the determinant and its infinitesimal trace (nb 4), and only at the
# end introduced coordinates — discovering that they merely re-encode what we already had, and that
# the plane's own scale-and-twist operators are the complex numbers (nb 5). For the same secret
# approached from the side of *groups* rather than *operators* — SO(2) ≅ U(1), ℂˣ, SU(2), and beyond —
# continue with the [`LieGroups/`](../LieGroups/README.md) series.
