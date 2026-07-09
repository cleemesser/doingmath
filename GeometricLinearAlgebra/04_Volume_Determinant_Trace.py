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
#     display_name: doingmath (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric Linear Algebra 4 — Volume, the Determinant, and a Coordinate-Free Trace
#
# Notebook 3 ended on a cliffhanger: a linear operator multiplies *every* signed area by one universal
# factor, $T(u)\wedge T(v) = (\det T)\,(u\wedge v)$. We now make that the **definition** of the
# determinant, lift it from area to **volume**, and extract from it the determinant's infinitesimal
# shadow — the **trace** — all without ever writing down a matrix.
#
# The plan:
# 1. **Volume element** = the top-degree wedge. In the plane it is area $u\wedge v$; in space it is the
#    signed volume $u\wedge v\wedge w$ (the scalar triple product).
# 2. **Determinant** = the factor by which an operator scales the volume element. One number, read off
#    geometrically; we confirm it equals the textbook determinant, that it multiplies under composition,
#    and that its sign is orientation and its vanishing is "collapse."
# 3. **Trace** = the *first-order* volume change: the rate at which volume grows if you start flowing
#    along $T$. Equivalently, the divergence of the linear flow $x\mapsto Tx$. We give a coordinate-free
#    formula, and close the loop with $\det(e^{tT}) = e^{t\,\operatorname{tr}T}$ — the exact bridge from
#    the infinitesimal (trace) to the finite (determinant) that the `LieGroups/` notebooks lean on.
#
# Figures use the shared `mathviz` library — 2D plots via matplotlib, 3D scenes via vedo.

# %% [markdown]
# May want to also add $\frac{d}{dt} det(I + t A) = tr(A)$ when evaluated at zero and also draw Arnold's picture for parallelogram.
# Also trace measures degree of self-mapping and the dimenionality of the subspace of a projection.

# %%
import numpy as np
import sympy as sp
sp.init_printing()
from scipy.linalg import expm
import mathviz as mv  # shared plane-viz library (see ../mathviz)

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
# ### Toolkit — now with a parallelepiped (the 3D volume box)


# %%
# Thin adapters over the shared `mathviz` library — see the note in notebook 3. new_plot returns a
# 2D mv.Plane (top_down) or a 3D mv.Space3D (top_down=False); the add_* helpers dispatch on the plot
# type. Filled parallelograms/parallelepipeds use mv's Polygon (2D) / Mesh3D (3D) primitives.
def new_plot(lim=3.0, top_down=True, axes=True, grid=True):
    if top_down:
        return mv.Plane(extent=lim, grid=grid, axes=axes)
    return mv.Space3D(bounds=lim)


def add_line(plot, pts, color, width=0.02, alpha=1.0):
    w = max(1.0, width * 130)
    if isinstance(plot, mv.Space3D):
        plot.line(pts, color=color, width=w, alpha=alpha)
    else:
        plot.curve(pts, color=color, width=w, alpha=alpha)
    return plot


def add_vector(plot, tail, head, color, label=None, label_size=0.7):
    tail = np.asarray(tail, float).reshape(-1)
    head = np.asarray(head, float).reshape(-1)
    if isinstance(plot, mv.Space3D):
        plot.arrow(tail, head, color=color)
    else:
        plot.vector(head - tail, origin=tail, color=color, label=label)
    return plot


def add_points(plot, pts, color, size=0.18):
    plot.points(np.atleast_2d(pts), color=color, size=max(5.0, size * 45))
    return plot


def add_parallelogram(plot, o, u, v, color, opacity=0.35, outline=True):
    plot.parallelogram(
        o, u, v, facecolor=color, alpha=opacity, edgecolor=(color if outline else None)
    )
    return plot


def add_parallelepiped(plot, o, u, v, w, color, opacity=0.18):
    """Filled, edged parallelepiped with corner o and edges u, v, w."""
    plot.parallelepiped(o, u, v, w, facecolor=color, alpha=opacity, edgecolor=color)
    return plot


# %% [markdown]
# ## 1. The volume element: the top-degree wedge
#
# The wedge keeps going. Wedging the largest number of vectors the space allows gives a single signed
# number — the **volume element**:
#
# - **Plane (2 arrows):** $u\wedge v$ = signed **area** (notebook 3).
# - **Space (3 arrows):** $u\wedge v\wedge w$ = signed **volume** of the parallelepiped they span.
#
# Concretely the 3D volume element is the **scalar triple product** $u\cdot(v\times w)$. It inherits
# the wedge's personality exactly: **multilinear** (linear in each arrow) and **alternating** (it
# vanishes whenever two arrows coincide — a flat, volumeless box), so swapping any two arrows flips
# the sign. The sign is the 3D orientation (right-handed +, left-handed −).


# %%
def cross3(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


def wedge3(u, v, w):
    """Signed volume of the parallelepiped spanned by u, v, w (scalar triple product)."""
    return float(np.dot(u, cross3(v, w)))


u = np.array([2.0, 0.3, 0.2])
v = np.array([0.4, 1.8, 0.1])
w = np.array([0.2, 0.5, 1.6])

print("signed volume u∧v∧w        :", round(wedge3(u, v, w), 6))
print("alternating  u∧u∧w = 0     :", np.isclose(wedge3(u, u, w), 0.0))
print("swap flips sign            :", np.isclose(wedge3(u, v, w), -wedge3(v, u, w)))
print(
    "multilinear (scale u by 3) :", np.isclose(wedge3(3 * u, v, w), 3 * wedge3(u, v, w))
)

p = new_plot(lim=3.0, top_down=False, axes=False)
add_parallelepiped(p, [0, 0, 0], u, v, w, BLUE)
add_vector(p, [0, 0, 0], u, BLUE, "u")
add_vector(p, [0, 0, 0], v, ORANGE, "v")
add_vector(p, [0, 0, 0], w, GREEN, "w")
p.display()
print(f"the box's signed volume is {wedge3(u, v, w):+.3f}")


# %% [markdown]
# ## 2. The determinant, defined by volume
#
# Apply an operator $T$ to every edge of the volume element. Multilinearity + the alternating property
# force the result to be the *same* element times a single scalar, no matter which edges you started
# with:
#
# $$T(u)\wedge T(v)\wedge T(w) = (\det T)\;(u\wedge v\wedge w).$$
#
# That scalar **is** the determinant — defined coordinate-free as **the factor by which $T$ scales
# signed volume**. Everything you know about determinants now reads off as geometry:
#
# - $|\det T|$ is the **volume-scaling factor**; its **sign** is whether $T$ preserves or flips
#   orientation.
# - $\det T = 0 \iff T$ **collapses** the volume element — the image is squashed into a lower
#   dimension, so $T$ is not invertible.
# - $\det(T\!\circ\!S) = \det T \cdot \det S$: scaling volume by $S$ then by $T$ multiplies the factors.
# - $\det(\text{rotation}) = +1$, $\det(\text{reflection}) = -1$, $\det(\text{shear}) = +1$,
#   $\det(\text{projection}) = 0$.
#
# We confirm the geometric definition reproduces `numpy.linalg.det`, then watch a unit cube turn into
# a parallelepiped whose volume is exactly $\det T$.


# %%
def det_by_volume(A):
    """Determinant as the volume-scaling factor: (Te1∧Te2∧Te3)/(e1∧e2∧e3), with e_i the unit box."""
    A = np.asarray(A, float)
    cols = [A[:, i] for i in range(A.shape[0])]
    if A.shape[0] == 2:
        return (
            cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0]
        )  # the 2D wedge of the columns
    return wedge3(*cols)  # the 3D wedge of the columns


rng = np.random.default_rng(4)
for n in (2, 3):
    M = rng.normal(size=(n, n))
    print(
        f"{n}D: det_by_volume = {det_by_volume(M): .6f}   np.linalg.det = {np.linalg.det(M): .6f}",
        "  match:",
        np.allclose(det_by_volume(M), np.linalg.det(M)),
    )

# Multiplicativity det(AB)=det(A)det(B), and that the volume factor is independent of the chosen edges.
A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
print(
    "det(AB) = det(A)·det(B):",
    np.allclose(det_by_volume(A @ B), det_by_volume(A) * det_by_volume(B)),
)
factors = []
for _ in range(5):  # T(u)∧T(v)∧T(w) / (u∧v∧w) is the same for any non-degenerate u,v,w
    uu, vv, ww = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    factors.append(wedge3(A @ uu, A @ vv, A @ ww) / wedge3(uu, vv, ww))
print(
    "volume factor independent of edges:",
    np.allclose(factors, det_by_volume(A)),
    "  (=",
    round(det_by_volume(A), 4),
    ")",
)

# %%
# A unit cube ↦ a parallelepiped; its volume equals det T.
T3 = np.array([[1.4, 0.3, -0.2], [0.1, 1.1, 0.4], [0.2, -0.3, 1.0]])
e1, e2, e3 = np.eye(3)
p = new_plot(lim=2.5, top_down=False, axes=False)
add_parallelepiped(p, [0, 0, 0], e1, e2, e3, FAINT, opacity=0.10)  # original unit cube
add_parallelepiped(p, [0, 0, 0], T3 @ e1, T3 @ e2, T3 @ e3, BLUE)  # its image
p.display()
print(f"unit cube (vol 1) ↦ parallelepiped of volume det T = {det_by_volume(T3):.4f}")


# %% [markdown]
# ## 3. Determinants of the notebook-2 operators
#
# A quick tour confirming the geometry: the operators we defined coordinate-free in notebook 2 have
# exactly the determinants their pictures demanded.

# %%
th = 0.6
ops2d = {
    "rotation θ": np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]),
    "uniform scale 1.6": 1.6 * np.eye(2),
    "shear k=0.8": np.array([[1, 0.8], [0, 1]]),
    "projection onto x": np.array([[1, 0], [0, 0]]),
    "reflection in x": np.array([[1, 0], [0, -1]]),
}
for name, M in ops2d.items():
    d = det_by_volume(M)
    tag = {
        1: "preserves area & orientation",
        -1: "flips orientation",
        0: "collapses to a line",
    }.get(round(d), "scales area")
    print(f"  {name:<20} det = {d:+.3f}   ({tag})")


# %% [markdown]
# ## 4. The trace: the determinant's infinitesimal shadow
#
# The determinant is the *finite* volume factor. Its **first-order** version is the **trace**. Nudge
# the identity a little in the direction of $T$ and ask how fast volume starts to change:
#
# $$\boxed{\;\operatorname{tr} T \;=\; \left.\frac{d}{dt}\right|_{t=0} \det(I + tT)\;}$$
#
# This is a coordinate-free definition: it never mentions a basis, only volumes. Differentiating the
# volume-element definition of $\det(I+tT)$ by the product rule gives an equally basis-free *formula* —
# replace one edge at a time by its $T$-image and add up:
#
# $$\operatorname{tr} T = \frac{(Tu)\wedge v\wedge w \;+\; u\wedge(Tv)\wedge w \;+\; u\wedge v\wedge(Tw)}{u\wedge v\wedge w}.$$
#
# Geometrically: the trace is the **total first-order stretch summed over independent directions** —
# and, as we will see, the **divergence** of the linear velocity field $x\mapsto Tx$.


# %%
def trace_by_derivative(A, t=1e-6):
    """tr A as the t→0 rate of change of det(I + tA)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    return (np.linalg.det(np.eye(n) + t * A) - 1.0) / t


def trace_by_wedge(A):
    """Coordinate-free trace: sum of volume elements with one edge replaced by its image (3D)."""
    A = np.asarray(A, float)
    u, v, w = np.eye(3)
    base = wedge3(u, v, w)
    return (wedge3(A @ u, v, w) + wedge3(u, A @ v, w) + wedge3(u, v, A @ w)) / base


M = rng.normal(size=(3, 3)).round(2) # generate a random 3x3 matrix.  substitute your own to play with it.
print("For matrix, M:")
sp.Matrix(M)


# %%
print("tr by derivative of det :", round(trace_by_derivative(M), 5))
print("tr by wedge formula     :", round(trace_by_wedge(M), 5))
print("np.trace (sum diagonal) :", round(np.trace(M), 5))
print(
    "all three agree         :",
    np.allclose(trace_by_derivative(M), np.trace(M))
    and np.allclose(trace_by_wedge(M), np.trace(M)),
)

# %% [markdown]
# ### The trace is the divergence of the linear flow
#
# Read $T$ as a **velocity field** $F(x) = Tx$: at each point $x$, move with velocity $Tx$. The
# divergence $\nabla\!\cdot\!F = \sum_i \partial F_i/\partial x_i$ measures the local rate of volume
# expansion of the flow — and for a linear field it is constant and equals $\operatorname{tr}T$. So a
# little ball of points, carried by the flow, initially inflates (or deflates) at the fractional rate
# $\operatorname{tr}T$ per unit time. We verify the divergence numerically by finite differences.


# %%
def divergence(F, x, h=1e-6):
    x = np.asarray(x, float)
    n = x.size
    d = 0.0
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        d += (F(x + e)[i] - F(x - e)[i]) / (2 * h)
    return d


F = lambda x: M @ x
pts = rng.normal(size=(5, 3))
divs = [divergence(F, x) for x in pts]
print("divergence of x↦Mx at 5 points:", np.round(divs, 4))
print("all equal tr M =", round(np.trace(M), 4), ":", np.allclose(divs, np.trace(M)))


# %% [markdown]
# ### Trace properties, and the bridge $\det e^{tT} = e^{t\,\operatorname{tr}T}$
#
# The trace is **linear** in $T$ and satisfies the **cyclic** identity $\operatorname{tr}(AB) =
# \operatorname{tr}(BA)$ — which is why (notebook 5) it will turn out to be *basis-independent* even
# though "sum of the diagonal" looks basis-bound. Finally, integrating the rate interpretation gives
# the clean exponential law connecting trace and determinant:
#
# $$\det\!\big(e^{tT}\big) = e^{\,t\,\operatorname{tr}T}.$$
#
# Volume under the flow grows exactly exponentially, with rate the trace — the finite determinant is
# the time-integral of the infinitesimal trace. (This is the identity behind "$\mathfrak{sl}_n$ =
# traceless ⇔ $SL_n$ = volume-preserving" in the Lie-group notebooks.)


# %%
A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
print(
    "trace is linear      :",
    np.allclose(np.trace(2 * A - 3 * B), 2 * np.trace(A) - 3 * np.trace(B)),
)
print("cyclic tr(AB)=tr(BA) :", np.allclose(np.trace(A @ B), np.trace(B @ A)))
for t in (0.3, 1.0, 2.5):
    lhs, rhs = np.linalg.det(expm(t * A)), np.exp(t * np.trace(A))
    print(
        f"  t={t}:  det(exp(tA)) = {lhs:.5f}   exp(t·trA) = {rhs:.5f}   match: {np.allclose(lhs, rhs)}"
    )


# %% [markdown]
# ## Summary
#
# - The **volume element** is the top-degree wedge: area $u\wedge v$ in the plane, signed volume
#   $u\wedge v\wedge w$ (the scalar triple product) in space — multilinear and alternating.
# - The **determinant** is the factor by which an operator scales that element:
#   $T(u)\wedge\cdots = (\det T)\,(u\wedge\cdots)$. Hence $|\det|$ = volume scaling, $\operatorname{sign}$
#   = orientation, $\det=0$ = collapse, and $\det(TS)=\det T\det S$ — all geometric, all confirmed
#   against `numpy`.
# - The **trace** is the determinant's first-order shadow: $\operatorname{tr}T = \frac{d}{dt}\big|_0\det(I+tT)$,
#   equivalently the **divergence** of the flow $x\mapsto Tx$, with the basis-free wedge formula and the
#   bridge $\det e^{tT}=e^{t\operatorname{tr}T}$.
# - Both are defined **without coordinates**; "product of eigenvalues" and "sum of the diagonal" are
#   *consequences*, recovered once we finally choose a basis.
#
# **Next (notebook 5):** we at last introduce **basis vectors**, the **coordinates** of a vector, and
# the **matrix** of an operator — recovering all of the above as familiar formulas — and discover that
# the operators which are "a scaling combined with a pure rotation" form a copy of the **complex
# numbers**.

# %% [markdown]
#
