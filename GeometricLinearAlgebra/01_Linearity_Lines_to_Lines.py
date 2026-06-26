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
# # Geometric Linear Algebra 1 — Linearity: Why Lines Go to Lines
#
# This is the first notebook in a series that builds linear algebra **geometrically** and, as far as
# possible, **coordinate-free**. We start with the only two things a vector can do — *add* and
# *scale* — and from those two operations alone we extract the single most important idea in the
# subject: **linearity**.
#
# A map $T$ is **linear** when it respects those two operations:
#
# $$T(u + v) = T(u) + T(v) \qquad\text{(additivity)}$$
# $$T(c\,v) = c\,T(v) \qquad\text{(homogeneity)}$$
#
# Everything else in this series — projection, rotation, scaling, shear, the dot and wedge products,
# area, volume, the determinant, the trace, and finally the discovery that the complex numbers *are*
# a kind of linear map — is a consequence of these two lines.
#
# The headline geometric fact we prove here: **a linear map sends straight lines to straight lines,
# sends parallel lines to parallel lines, and fixes the origin.** That is what makes linear algebra
# *the geometry of flat space*.
#
# > **Visualization.** This series uses **`k3d`**, an interactive WebGL renderer, for every picture.
# > We are doing plane geometry now, so we draw everything in the $z=0$ plane and look straight down
# > at it — but because the canvas is genuinely 3D, the *same* tools will carry us into space in the
# > later notebooks with nothing to relearn. Drag to orbit, scroll to zoom; the figures are live.
#
# > **A note on coordinates.** We are doing geometry, so a vector means "an arrow / a displacement,"
# > not "a list of numbers." We have *no basis yet* — coordinates and matrices are not introduced
# > until notebook 5. To draw a picture we are forced to put numbers on the page, so treat the pairs
# > `(x, y)` purely as **plotting coordinates for the renderer**; the statements we make ("this map
# > is linear," "lines go to lines") are about the arrows themselves, never about the numbers.

# %%
import numpy as np
import k3d

# Color palette shared across the whole series (k3d wants integer hex colors).
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
# ### A tiny k3d toolkit
#
# A handful of helpers we reuse below (and throughout the series). Each lifts our 2D arrows and lines
# into the $z=0$ plane and hands them to k3d. `new_plot()` returns a dark-themed scene; by default it
# parks the camera directly overhead so plane geometry reads as a flat top-down picture.


# %%
def _to3(pts):
    """Lift (...,2) plane points into (...,3) by appending z=0; pass (...,3) through."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[-1] == 3:
        return pts
    zeros = np.zeros(pts.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate([pts, zeros], axis=-1)


def new_plot(lim=3.0, top_down=True, axes=True, grid=True):
    """A dark k3d scene. top_down parks the camera overhead for plane (2D) geometry."""
    plot = k3d.plot(
        background_color=BG,
        grid_color=GRIDC,
        label_color=LABELC,
        grid_visible=grid,
        camera_auto_fit=not top_down,
        menu_visibility=False,
    )
    if top_down:
        # camera = [eye_xyz, target_xyz, up_xyz]; look straight down the +z axis.
        plot.camera = [0, 0, 2.6 * lim, 0, 0, 0, 0, 1, 0]
    if axes:
        add_line(plot, np.array([[-lim, 0], [lim, 0]]), GREY, width=0.012)
        add_line(plot, np.array([[0, -lim], [0, lim]]), GREY, width=0.012)
    return plot


def add_line(plot, pts, color, width=0.02, alpha=1.0):
    """Add a polyline through the (N,2 or 3) points."""
    plot += k3d.line(_to3(pts), color=color, width=width, shader="thick", opacity=alpha)


def add_vector(plot, tail, head, color, label=None, label_size=0.7):
    """Draw an arrow from tail to head (2D or 3D), optionally labeled near its tip."""
    tail3 = _to3(np.atleast_2d(tail))
    head3 = _to3(np.atleast_2d(head))
    plot += k3d.vectors(
        origins=tail3,
        vectors=(head3 - tail3),
        color=color,
        head_size=1.5,
        line_width=0.03,
    )
    if label:
        pos = (tail3[0] + 0.55 * (head3[0] - tail3[0])).tolist()
        plot += k3d.text(label, position=pos, color=color, size=label_size, label_box=False)


def add_points(plot, pts, color, size=0.18):
    plot += k3d.points(_to3(np.atleast_2d(pts)), color=color, point_size=size, shader="3d")


# %% [markdown]
# ## 1. A vector is an arrow: the two operations
#
# Before any map can be *linear*, we need the two operations linearity is about. A **vector** is a
# displacement — an arrow with a length and a direction, free to slide anywhere (its tail is not
# pinned). There are exactly two things we can do with arrows:
#
# 1. **Add** them, tip-to-tail: $u + v$ is "walk along $u$, then along $v$."
# 2. **Scale** them by a number: $c\,v$ stretches $v$ by the factor $c$ (and flips it if $c<0$).
#
# A **linear combination** $a\,u + b\,v$ is the result of doing both. The whole of flat geometry is
# generated by these moves, so a map that *commutes with both* preserves the geometry. Let us draw
# the two operations once, to fix the picture. (Drag to orbit — you are looking down at the plane.)

# %%
u = np.array([2.0, 0.6])
v = np.array([0.7, 1.8])

p = new_plot(lim=3.5)
# Addition as tip-to-tail and as a parallelogram.
add_vector(p, [0, 0], u, BLUE, "u")
add_vector(p, u, u + v, ORANGE, "v")
add_vector(p, [0, 0], u + v, GREEN, "u+v")
add_vector(p, [0, 0], v, FAINT)  # v from origin (faint side of parallelogram)
add_vector(p, v, u + v, FAINT)  # closing side
p.display()

# %%
# Scaling lives on a single line through the origin: stretch (c>1), shrink (0<c<1), flip (c<0).
p = new_plot(lim=3.5)
for c, col, lab in [(1.5, GREEN, "1.5·u"), (1.0, BLUE, "u"), (-0.8, RED, "-0.8·u")]:
    add_vector(p, [0, 0], c * u, col, lab)
p.display()

# %% [markdown]
# Numerically, the two operations are exactly what you expect — but notice we are only *adding* and
# *scaling*; no other operation on arrows is allowed in a vector space.

# %%
print("u           =", u)
print("v           =", v)
print("u + v       =", u + v)
print("1.5 · u     =", 1.5 * u)
print("addition is commutative:", np.allclose(u + v, v + u))


# %% [markdown]
# ## 2. What "linear" means
#
# A map $T$ takes vectors to vectors. It is **linear** if it does not care whether you combine
# *before* or *after* applying it:
#
# $$T(a\,u + b\,v) = a\,T(u) + b\,T(v).$$
#
# That single equation packages both required properties (set $a=b=1$ to get additivity; set $b=0$
# to get homogeneity). Two immediate consequences fall straight out of it:
#
# - **The origin is fixed.** Put $c=0$ in homogeneity: $T(0) = T(0\cdot v) = 0\,T(v) = 0$. A linear
#   map can never move the origin.
# - **Linear combinations are preserved.** By induction, $T\!\big(\sum_i c_i v_i\big) = \sum_i c_i\,T(v_i)$.
#   This is the property we will lean on for everything that follows.
#
# To *experiment* we need some concrete linear maps. The cleanest way to manufacture one is to pick
# where two reference arrows go and extend "by linearity." We package that as a small operator
# object. Its internals use a $2\times2$ array purely as bookkeeping — we are **not** yet calling
# that a "matrix" or introducing a basis; it is just a convenient way to store a linear action so we
# can apply it to many arrows at once.


# %%
class LinearMap:
    """A linear operator on the plane, stored by its action. NOT yet 'a matrix with a basis' —
    that interpretation waits until notebook 5. Here it is simply a callable that we have *verified*
    satisfies additivity and homogeneity."""

    def __init__(self, action):
        self._A = np.asarray(action, float)  # 2x2 bookkeeping array

    def __call__(self, v):
        v = np.asarray(v, float)
        return v @ self._A.T  # apply to a single vector or a stack of row-vectors


# A linear map that takes e1 ↦ (1, 0.5) and e2 ↦ (-0.4, 1.2), extended by linearity.
T = LinearMap([[1.0, -0.4], [0.5, 1.2]])

# An *affine* map  x ↦ T(x) + b  — emphatically NOT linear (it moves the origin).
b = np.array([1.3, -0.7])


def affine(v):
    return T(v) + b


# A genuinely *nonlinear* map that bends straight lines.
def bend(v):
    v = np.asarray(v, float)
    x, y = v[..., 0], v[..., 1]
    return np.stack([x, y + 0.35 * x**2], axis=-1)


# %% [markdown]
# ## 3. The 1D case: a linear map of the line is "multiply by a number"
#
# Strip away the second dimension and linearity becomes startlingly rigid. On the line, every vector
# is a scalar multiple of one unit arrow, $v = x\cdot e$. So by homogeneity a linear map is pinned
# down by what it does to that single arrow:
#
# $$T(x\,e) = x\,T(e) = x\,(a\,e) = (a x)\,e, \qquad a := \text{the stretch factor.}$$
#
# **Every linear map of the line is multiplication by a constant $a$** — a pure scaling. That is the
# entire story in 1D, and it is why "slope-intercept" lines $x\mapsto ax+b$ are *not* linear maps
# unless $b=0$: the intercept moves the origin. We plot each map as the graph $y=T(x)$ (a line in the
# $z=0$ plane); the linear one passes through the origin, the affine one does not.

# %%
def lin1d(x, a=1.7):
    return a * x


def affine1d(x, a=1.7, b=0.8):
    return a * x + b


# Linearity test on the line.
pp, qq, cc = 0.9, -1.3, 2.4
print("LINEAR  x↦1.7x")
print("  additivity   T(p+q)=T(p)+T(q):", np.allclose(lin1d(pp + qq), lin1d(pp) + lin1d(qq)))
print("  homogeneity  T(c·p)=c·T(p)   :", np.allclose(lin1d(cc * pp), cc * lin1d(pp)))
print("AFFINE  x↦1.7x+0.8  (intercept ≠ 0)")
print("  additivity   T(p+q)=T(p)+T(q):", np.allclose(affine1d(pp + qq), affine1d(pp) + affine1d(qq)))
print("  fixes origin T(0)=0          :", np.allclose(affine1d(0.0), 0.0))

xx = np.linspace(-2, 2, 50)
p = new_plot(lim=4.0)
add_line(p, np.stack([xx, lin1d(xx)], axis=1), BLUE, width=0.04)  # linear: through origin
add_line(p, np.stack([xx, affine1d(xx)], axis=1), ORANGE, width=0.04)  # affine: offset
add_points(p, [0, 0], GREEN, size=0.22)  # origin: on the blue line only
p.display()
print("\nblue = linear x↦1.7x (through origin) · orange = affine x↦1.7x+0.8 · green dot = origin")


# %% [markdown]
# ## 4. The headline theorem: lines go to lines
#
# Now the plane. A straight line is the set of points you reach by starting at some point $p_0$ and
# walking along a fixed direction $d$:
#
# $$\ell(t) = p_0 + t\,d, \qquad t \in \mathbb{R}.$$
#
# Apply a linear map $T$ and use linearity *directly*:
#
# $$T(\ell(t)) = T(p_0 + t\,d) = T(p_0) + t\,T(d).$$
#
# The right-hand side is **again the parametric equation of a line** — it starts at the image point
# $T(p_0)$ and runs along the image direction $T(d)$. So the image of a line is a line. (The one
# degenerate case: if $T(d) = 0$ the whole line collapses to the single point $T(p_0)$. Lines never
# *bend* — at worst they are crushed.)
#
# Two corollaries come for free from the same computation:
#
# - **Parallel lines stay parallel.** Two lines are parallel iff they share the direction $d$; their
#   images share the direction $T(d)$, so they stay parallel (or coincide).
# - **Equally-spaced points stay equally spaced** along the image line, because the parameter $t$
#   rides along untouched. Midpoints map to midpoints.
#
# Let us *watch* it: each scene shows the original square grid (faint) and its image (bold) overlaid.


# %%
def grid_lines(n=11, lim=2.0, samples=40):
    """A square grid of straight lines, as a list of (samples,2) polylines."""
    ticks = np.linspace(-lim, lim, n)
    s = np.linspace(-lim, lim, samples)
    lines = []
    for tk in ticks:
        lines.append(np.stack([s, np.full_like(s, tk)], axis=1))  # horizontal
        lines.append(np.stack([np.full_like(s, tk), s], axis=1))  # vertical
    return lines


def show_transform(func, color, lim=2.0, title=""):
    """Overlay the original grid (faint) with its image under func (bold), plus origin markers."""
    p = new_plot(lim=2.4 * lim, axes=True)
    for ln in grid_lines(lim=lim):
        add_line(p, ln, FAINT, width=0.012)  # original grid, faint
        add_line(p, func(ln), color, width=0.02)  # image grid, bold
    add_points(p, [0, 0], GREY, size=0.16)  # original origin
    add_points(p, func(np.array([0.0, 0.0])), GREEN, size=0.24)  # image of origin
    if title:
        print(title)
    p.display()


# Linear T: the grid shears/stretches but the origin (green) stays put.
show_transform(T.__call__, BLUE, title="LINEAR T — lines→lines, parallels stay parallel, ORIGIN FIXED (green on grey)")

# %%
# Affine T+b: still lines→lines, parallels still parallel — but the green origin has MOVED off grey.
show_transform(affine, ORANGE, title="AFFINE T(x)+b — still flat, but the ORIGIN MOVED (green ≠ grey)")

# %% [markdown]
# Both maps keep every line straight and keep parallels parallel — that is the *flatness* of linear
# and affine maps. The difference is the green dot: the linear map pins the origin; the affine map
# slides the whole grid off it. (Affine = linear + a translation. We focus on the linear part because
# the translation carries no *shape* information — it just relocates.)

# %% [markdown]
# ### Verifying the theorem numerically, not just by eye
#
# A picture can mislead. Let us *prove* that $T$ keeps a line straight by checking the algebra:
# sample three points on a line and confirm their images are still **collinear**. Collinearity of
# $A, B, C$ means the displacement $B-A$ is parallel to $C-A$, i.e. their 2D cross-product (the
# scalar $x_1 y_2 - x_2 y_1$, which returns as the *wedge* in notebook 3) vanishes.

# %%
def cross2(a, b):
    """Scalar 2D cross product — zero exactly when a, b are parallel."""
    return a[0] * b[1] - a[1] * b[0]


def collinear(A, B, C, tol=1e-9):
    return abs(cross2(B - A, C - A)) < tol


p0 = np.array([-1.3, 0.4])
d = np.array([0.8, 1.1])
A, B, C = p0, p0 + 0.37 * d, p0 + 0.91 * d  # three unequal parameter values

print("LINEAR  T")
print("  pre-image collinear:", collinear(A, B, C))
print("  image    collinear:", collinear(T(A), T(B), T(C)))
print("BEND  (nonlinear)")
print("  pre-image collinear:", collinear(A, B, C))
print("  image    collinear:", collinear(bend(A), bend(B), bend(C)), "  ← line got bent!")

M = 0.5 * (A + C)
print("\nmidpoint preserved by T:", np.allclose(T(M), 0.5 * (T(A) + T(C))))


# %% [markdown]
# ## 5. What breaks linearity: bending the grid
#
# To appreciate that "lines → lines" is a real constraint and not automatic, watch a **nonlinear**
# map. Our `bend` map nudges each point upward by an amount proportional to $x^2$. It is perfectly
# smooth and continuous — but it is not linear, and the symptom is unmistakable: the straight grid
# lines come out **curved**.

# %%
show_transform(bend, RED, title="NONLINEAR x↦(x, y+0.35x²) — straight grid lines bend into parabolas")


# %% [markdown]
# ## 6. The two laws, tested head-to-head
#
# Finally, the definition itself. We throw random vectors and random scalars at each map and check
# additivity and homogeneity. The linear map passes every time; the affine and nonlinear maps fail.
# This is the *operational* meaning of linearity — no pictures required.

# %%
rng = np.random.default_rng(0)


def check_linearity(func, name, trials=2000):
    add_ok = hom_ok = True
    for _ in range(trials):
        u_, v_ = rng.normal(size=2), rng.normal(size=2)
        c_ = rng.normal()
        add_ok &= np.allclose(func(u_ + v_), func(u_) + func(v_))
        hom_ok &= np.allclose(func(c_ * u_), c_ * func(u_))
    origin_ok = np.allclose(func(np.zeros(2)), np.zeros(2))
    print(f"{name:<24} additivity={str(add_ok):<5} homogeneity={str(hom_ok):<5} fixes_origin={origin_ok}")


print(f"{'map':<24} is it linear?")
print("-" * 66)
check_linearity(T, "T  (linear)")
check_linearity(affine, "T(x)+b  (affine)")
check_linearity(bend, "bend  (nonlinear)")

# %% [markdown]
# As predicted: only `T` satisfies both laws and fixes the origin. The affine map fails additivity
# and homogeneity (the translation `b` leaks through) and moves the origin; the nonlinear map fails
# because of the $x^2$ term.

# %% [markdown]
# ## Summary
#
# - A **vector** is an arrow you can **add** (tip-to-tail) and **scale**. Those two operations are
#   the entire raw material of linear algebra.
# - A map is **linear** iff $T(au + bv) = a\,T(u) + b\,T(v)$. Equivalently: additive + homogeneous.
# - Consequences, proved straight from the definition:
#   - $T(0) = 0$ — **the origin is fixed**.
#   - **Lines go to lines**: $T(p_0 + t\,d) = T(p_0) + t\,T(d)$ is again a line (or a point).
#   - **Parallel lines stay parallel**, and **midpoints map to midpoints**.
# - In **1D**, every linear map is just multiplication by a constant — a pure scaling. Slope-intercept
#   "lines" $ax+b$ are *affine*, not linear, the moment $b \neq 0$.
# - **Affine = linear + translation.** The translation only relocates; all the *shape* of the map
#   lives in its linear part, which is what the rest of the series studies.
#
# **Next (notebook 2):** we stop *manufacturing* linear maps and start *defining the important ones
# geometrically* — projection onto a vector, rotation, scaling, and shear — entirely without
# coordinates, using only lengths and the right-angle relationship between arrows.
