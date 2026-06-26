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
# # Geometric Linear Algebra 2 — The Operators, Defined Without Coordinates
#
# In notebook 1 we *manufactured* a linear map by declaring where two arrows go. Now we do the
# opposite: we take the geometric operations that actually matter — **scaling**, **projection onto a
# vector**, **rotation**, **shear** (and **reflection** as a bonus) — and give each a definition that
# names *what it does to arrows*, with **no basis and no matrix in sight**. Then we verify, by
# computation, that each one is linear and has the geometric properties we claimed.
#
# To talk about projection and rotation we need to *measure* arrows — their lengths and the angles
# between them. That measuring tool is the **dot product**. We introduce just enough of it here to
# build the operators; notebook 3 develops it in full alongside its antisymmetric twin, the wedge.
#
# Two threads to watch:
# - **Composition.** Operators are maps, so they compose — and the order usually matters. A scaling
#   then a rotation is a "scale-and-rotate"; that particular combination is the seed of the complex
#   numbers (notebook 5).
# - **The 90° rotation $J$.** Rotation by any angle turns out to be $\cos\theta\,I + \sin\theta\,J$,
#   where $J$ is the quarter-turn. We will find $J^2 = -I$ — the geometric origin of $i^2=-1$.
#
# As before, everything is drawn with **k3d** in the $z=0$ plane (drag to orbit, scroll to zoom).

# %%
import numpy as np
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
# ### The k3d toolkit (same as notebook 1) plus a test shape
#
# We reuse the plotting helpers, and add a small asymmetric **"flag" polygon** as a probe: an
# asymmetric shape makes rotations, shears, and reflections instantly legible (you can see if it was
# turned, slanted, or mirrored).


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


# An asymmetric closed polygon ("flag on a pole") used to make each transform visible.
FLAG = np.array(
    [
        [0.0, 0.0], [0.0, 2.0], [1.2, 1.6], [0.6, 1.2], [0.0, 1.2], [0.0, 0.0],
    ]
)


def draw_shape(plot, pts, color, width=0.03):
    add_line(plot, pts, color, width=width)


def grid_lines(n=9, lim=2.0, samples=30):
    ticks = np.linspace(-lim, lim, n)
    s = np.linspace(-lim, lim, samples)
    lines = []
    for tk in ticks:
        lines.append(np.stack([s, np.full_like(s, tk)], axis=1))
        lines.append(np.stack([np.full_like(s, tk), s], axis=1))
    return lines


def show_operator(func, color, lim=2.0, shape=True, gridded=True):
    """Overlay the original (faint) with its image (bold): grid and/or the flag shape."""
    p = new_plot(lim=2.6 * lim, axes=True)
    if gridded:
        for ln in grid_lines(lim=lim):
            add_line(p, ln, FAINT, width=0.01)
            add_line(p, func(ln), color, width=0.018)
    if shape:
        draw_shape(p, FLAG, FAINT, width=0.02)
        draw_shape(p, func(FLAG), color, width=0.04)
    add_points(p, [0, 0], GREEN, size=0.16)
    return p


# %% [markdown]
# ## 1. The dot product: our ruler and protractor
#
# Everything metric (length, angle, perpendicularity) comes from one bilinear gadget, the **dot
# product**. Geometrically it is
#
# $$\langle u, v\rangle = |u|\,|v|\,\cos\theta,$$
#
# the product of the two lengths with the cosine of the angle between them. From it we read off:
#
# - **length**: $|v| = \sqrt{\langle v,v\rangle}$ (since $\theta=0$ with itself),
# - **angle**: $\cos\theta = \langle u,v\rangle / (|u|\,|v|)$,
# - **perpendicularity**: $u \perp v \iff \langle u,v\rangle = 0$.
#
# We also need the **quarter-turn** $J$: the operator that rotates every arrow $90°$ counter-clockwise.
# Geometrically that is all the definition we need; concretely it sends the arrow with plane
# coordinates $(x,y)$ to $(-y,\,x)$. It will be the workhorse of projection and rotation.


# %%
def dot(u, v):
    """Dot product, summing over the last axis (works on single vectors or stacks)."""
    return np.sum(np.asarray(u, float) * np.asarray(v, float), axis=-1)


def norm(v):
    return np.sqrt(dot(v, v))


def angle_between(u, v):
    return np.arccos(np.clip(dot(u, v) / (norm(u) * norm(v)), -1.0, 1.0))


def J(v):
    """Quarter-turn: rotate every arrow 90° counter-clockwise. (x,y) ↦ (-y, x)."""
    v = np.asarray(v, float)
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


# Sanity: J turns an arrow by exactly 90°, preserves its length, and J∘J = −identity.
a = np.array([2.0, 0.7])
print("J is a 90° turn :", np.allclose(angle_between(a, J(a)), np.pi / 2))
print("J preserves len :", np.allclose(norm(J(a)), norm(a)))
print("J∘J = −I        :", np.allclose(J(J(a)), -a), "   ← the geometric seed of i² = −1")


# %% [markdown]
# ## 2. Scaling — the simplest operator
#
# **Uniform scaling** stretches every arrow by the same factor $\lambda$ about the origin:
#
# $$S_\lambda(v) = \lambda\,v.$$
#
# Linearity is immediate ($S_\lambda(au+bv)=\lambda(au+bv)=a S_\lambda u + b S_\lambda v$). It scales
# every length by $|\lambda|$ and (we will see in notebook 4) every area by $\lambda^2$.
#
# **Directional scaling** stretches by $\lambda$ along *one* direction $a$ only, leaving the
# perpendicular direction fixed. We can write it coordinate-free using the part of $v$ lying along
# the unit direction $\hat a$:
#
# $$S_{a,\lambda}(v) = v + (\lambda - 1)\,\langle v,\hat a\rangle\,\hat a.$$


# %%
def scale_uniform(v, lam):
    return lam * np.asarray(v, float)


def scale_dir(v, a, lam):
    """Scale by lam along direction a, identity perpendicular to a."""
    v = np.asarray(v, float)
    ahat = np.asarray(a, float) / norm(a)
    comp = dot(v, ahat)[..., None] * ahat  # part of v along a
    return v + (lam - 1.0) * comp


show_operator(lambda v: scale_uniform(v, 1.6), BLUE).display()
print("uniform scale ×1.6 — every length grows by 1.6, the shape keeps its form")

# %%
show_operator(lambda v: scale_dir(v, [1.0, 0.0], 2.0), ORANGE).display()
print("directional scale ×2 along x — horizontal stretches, vertical untouched")

# %% [markdown]
# Both are linear (we verify all operators together in §7). Uniform scaling preserves *shape* (angles
# unchanged); directional scaling distorts it.

# %% [markdown]
# ## 3. Projection onto a vector
#
# **Project $v$ onto the line through $a$**: drop a perpendicular from the tip of $v$ to that line.
# The foot of that perpendicular is the part of $v$ that "lies along $a$." Using the dot product to
# measure that component:
#
# $$P_a(v) = \frac{\langle v, a\rangle}{\langle a, a\rangle}\,a.$$
#
# Three properties make this *the* projection, all checkable:
#
# 1. **Idempotent**: $P_a(P_a(v)) = P_a(v)$ — projecting twice is the same as once (you are already on
#    the line).
# 2. **Fixes the line**: $P_a(a) = a$, and more generally $P_a(c\,a)=c\,a$.
# 3. **Kills the perpendicular**: if $v \perp a$ then $P_a(v) = 0$.
#
# The leftover piece $v - P_a(v)$ is the **rejection** — the projection onto the perpendicular
# direction. Together they split *any* arrow into "along $a$" + "perpendicular to $a$."


# %%
def project(v, a):
    """Orthogonal projection of v onto the line spanned by a."""
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    return (dot(v, a) / dot(a, a))[..., None] * a


def reject(v, a):
    """The perpendicular leftover: v split into along-a plus this."""
    return np.asarray(v, float) - project(v, a)


a = np.array([1.0, 0.5])
v = np.array([0.4, 1.8])
pv, rv = project(v, a), reject(v, a)

print("idempotent  P(P v)=P v :", np.allclose(project(pv, a), pv))
print("fixes the line P(a)=a  :", np.allclose(project(a, a), a))
print("kills perp  P(J a)=0   :", np.allclose(project(J(a), a), np.zeros(2)))
print("split  Pv + (v−Pv)=v   :", np.allclose(pv + rv, v))
print("rejection ⟂ a          :", np.allclose(dot(rv, a), 0.0))

# Picture: v, its shadow Pv on the line through a, and the perpendicular rejection.
p = new_plot(lim=3.0)
add_line(p, np.array([-2.5 * a, 2.5 * a]), GREY, width=0.012)  # the line through a
add_vector(p, [0, 0], a, PURPLE, "a")
add_vector(p, [0, 0], v, BLUE, "v")
add_vector(p, [0, 0], pv, GREEN, "P v")
add_vector(p, pv, v, RED, "v − P v")  # the dropped perpendicular
p.display()

# %%
show_operator(lambda w: project(w, a), GREEN, shape=True).display()
print("projection collapses the whole plane (and the flag) onto the line through a")

# %% [markdown]
# Notice the projection is **not invertible** — the entire plane is crushed onto a line, so
# information is lost. In notebook 4 this shows up as *zero area scaling*: a determinant of $0$.

# %% [markdown]
# ## 4. Rotation
#
# A **rotation by angle $\theta$** is the linear map that turns every arrow through $\theta$ while
# preserving all lengths and the orientation of the plane. We can build it coordinate-free from the
# quarter-turn $J$. Split the rotated image of $v$ into a part along $v$ and a part along $Jv$
# (its $90°$ partner): turning $v$ by $\theta$ keeps a $\cos\theta$ share along $v$ and adds a
# $\sin\theta$ share along $Jv$:
#
# $$R_\theta(v) = \cos\theta\;v + \sin\theta\;J v = (\cos\theta\,I + \sin\theta\,J)\,v.$$
#
# This single formula is loaded with future meaning. Reading $I$ as "$1$" and $J$ as "$i$" (legal,
# since $J^2=-I$ mirrors $i^2=-1$), it says $R_\theta = \cos\theta + i\sin\theta = e^{i\theta}$ — the
# complex exponential, discovered as a rotation operator. We pick that thread up properly in notebook 5.
#
# Rotations are **orthogonal** (they preserve every dot product, hence all lengths and angles) and
# they **compose by adding angles**: $R_\alpha R_\beta = R_{\alpha+\beta}$ — which is exactly the
# angle-sum identities for sine and cosine, falling out for free.


# %%
def rotate(v, theta):
    v = np.asarray(v, float)
    return np.cos(theta) * v + np.sin(theta) * J(v)


th, ph = 0.7, 1.1
u_, w_ = np.array([1.3, -0.4]), np.array([0.2, 1.7])

print("preserves length      :", np.allclose(norm(rotate(u_, th)), norm(u_)))
print("preserves dot/angle    :", np.allclose(dot(rotate(u_, th), rotate(w_, th)), dot(u_, w_)))
print("turns by exactly θ     :", np.allclose(angle_between(u_, rotate(u_, th)), th))
print("R(α)R(β)=R(α+β)        :", np.allclose(rotate(rotate(u_, ph), th), rotate(u_, th + ph)))
print("rotations commute      :", np.allclose(rotate(rotate(u_, ph), th), rotate(rotate(u_, th), ph)))

show_operator(lambda v: rotate(v, np.pi / 5), BLUE).display()
print("rotation by 36° — lengths and angles preserved, orientation preserved, origin fixed")


# %% [markdown]
# ## 5. Shear
#
# A **shear** (transvection) fixes a whole line and slides everything *parallel* to that line by an
# amount proportional to the distance from it — think of a deck of cards pushed sideways. Pick a unit
# direction $d$ for the shear line and let $n = Jd$ be its unit normal. The shear of strength $k$ is
#
# $$H(v) = v + k\,\langle v, n\rangle\,d.$$
#
# It is linear; it **fixes every arrow along $d$** (those have $\langle v,n\rangle=0$); and, crucially,
# it **preserves area** — it slides without stretching. (In notebook 4 that is a determinant of
# exactly $1$.) Shear is the canonical example of a linear map that is neither a scaling nor a
# rotation, yet still sends lines to lines.


# %%
def shear(v, d, k):
    v = np.asarray(v, float)
    d = np.asarray(d, float) / norm(d)
    n = J(d)  # unit normal to the shear line
    return v + k * dot(v, n)[..., None] * d


def signed_area(P, Q):
    """Signed area of the parallelogram spanned by P and Q (the 2D cross product)."""
    return P[..., 0] * Q[..., 1] - P[..., 1] * Q[..., 0]


d = np.array([1.0, 0.0])
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
He1, He2 = shear(e1, d, 0.8), shear(e2, d, 0.8)
print("fixes the shear line  :", np.allclose(shear(d, d, 0.8), d))
print("preserves area (det=1):", np.allclose(signed_area(He1, He2), signed_area(e1, e2)))

show_operator(lambda v: shear(v, d, 0.8), ORANGE).display()
print("horizontal shear k=0.8 — vertical lines tilt, the x-axis stays fixed, areas unchanged")


# %% [markdown]
# ## 6. Reflection (bonus)
#
# A **reflection across the line through $a$** flips the perpendicular component while keeping the
# along-$a$ component — it is "twice the projection, minus the original":
#
# $$F_a(v) = 2\,P_a(v) - v.$$
#
# It preserves lengths (like a rotation) but **reverses orientation** (the flag comes out mirrored).
# In notebook 4 that is a determinant of $-1$.


# %%
def reflect(v, a):
    return 2.0 * project(v, a) - np.asarray(v, float)


a = np.array([1.0, 0.4])
print("preserves length:", np.allclose(norm(reflect(v, a)), norm(v)))
print("involution F∘F=I:", np.allclose(reflect(reflect(v, a), a), v))
show_operator(lambda w: reflect(w, a), PURPLE).display()
print("reflection across the line through a — the flag is mirrored (orientation flips)")


# %% [markdown]
# ## 7. They are all linear — and composition is where the order matters
#
# First, the unifying fact: **every operator above is linear**. We test additivity and homogeneity on
# random inputs, exactly as in notebook 1.

# %%
rng = np.random.default_rng(1)
a = np.array([1.0, 0.5])
d = np.array([1.0, 0.0])

OPERATORS = {
    "uniform scale ×1.6": lambda v: scale_uniform(v, 1.6),
    "dir. scale ×2 / x": lambda v: scale_dir(v, [1, 0], 2.0),
    "projection onto a": lambda v: project(v, a),
    "rotation by 36°": lambda v: rotate(v, np.pi / 5),
    "shear k=0.8": lambda v: shear(v, d, 0.8),
    "reflection in a": lambda v: reflect(v, a),
}

for name, f in OPERATORS.items():
    add_ok = hom_ok = True
    for _ in range(1500):
        u1, u2, c = rng.normal(size=2), rng.normal(size=2), rng.normal()
        add_ok &= np.allclose(f(u1 + u2), f(u1) + f(u2))
        hom_ok &= np.allclose(f(c * u1), c * f(u1))
    print(f"{name:<20} linear: additivity={add_ok}  homogeneity={hom_ok}  fixes_origin={np.allclose(f(np.zeros(2)), 0)}")

# %% [markdown]
# Now composition. Operators are maps, so $\,(g\circ f)(v) = g(f(v))$ is again linear — but reversing
# the order generally changes the result. We show two things at once: a **scale-then-rotate** (the
# "scale-and-twist" that becomes a complex number in notebook 5), and the fact that **rotate-then-shear
# $\neq$ shear-then-rotate**.

# %%
def rotate_then_scale(v):  # first scale by 1.5, then rotate by 50°
    return rotate(scale_uniform(v, 1.5), np.deg2rad(50))


vtest = np.array([1.0, 0.3])
rs = rotate(shear(vtest, d, 0.8), np.pi / 5)
sr = shear(rotate(vtest, np.pi / 5), d, 0.8)
print("scale∘rotate is linear   :", np.allclose(rotate_then_scale(2 * vtest), 2 * rotate_then_scale(vtest)))
print("rotate∘shear == shear∘rotate ?", np.allclose(rs, sr), "  ← order matters!")
print("   rotate(shear v) =", np.round(rs, 3))
print("   shear(rotate v) =", np.round(sr, 3))

show_operator(rotate_then_scale, GREEN).display()
print("scale ×1.5 then rotate 50° — a 'scale-and-twist'; this composite is a complex number (nb 5)")


# %% [markdown]
# ## Summary
#
# Every operator was defined by **what it does to arrows**, using only lengths, perpendiculars, and
# the quarter-turn $J$ — never a matrix:
#
# | operator | coordinate-free definition | preserves | notebook-4 determinant |
# |----------|---------------------------|-----------|------------------------|
# | uniform scale | $S_\lambda(v)=\lambda v$ | shape (angles) | $\lambda^2$ |
# | directional scale | $v+(\lambda-1)\langle v,\hat a\rangle\hat a$ | the $\perp$ direction | $\lambda$ |
# | projection | $\frac{\langle v,a\rangle}{\langle a,a\rangle}a$ | the line through $a$ | $0$ (crushes the plane) |
# | rotation | $\cos\theta\,v+\sin\theta\,Jv$ | lengths, angles, orientation | $+1$ |
# | shear | $v+k\langle v,n\rangle d$ | the shear line, **area** | $+1$ |
# | reflection | $2P_a(v)-v$ | lengths; **reverses** orientation | $-1$ |
#
# - Metric operators (projection, rotation, reflection) are built from the **dot product**; shear and
#   scaling are more primitive.
# - The quarter-turn satisfies $J^2=-I$, and $R_\theta=\cos\theta\,I+\sin\theta\,J$ — the rotation
#   operator is already "$e^{i\theta}$" in disguise.
# - Operators **compose**, and composition is generally **non-commutative**.
#
# **Next (notebook 3):** we develop the two bilinear measuring tools properly — the **dot product**
# (symmetric: lengths and angles) and the **wedge product** (antisymmetric: signed area) — and show
# they are the symmetric and antisymmetric halves of "multiplying two vectors."
