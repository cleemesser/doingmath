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
# # Geometric Linear Algebra 3 — Dot and Wedge: Lengths, Angles, and Signed Area
#
# We used the dot product in notebook 2 as a black-box ruler. Now we earn it, and we meet its
# antisymmetric twin. There are exactly **two** ways to combine a pair of arrows into a single
# bilinear quantity, and together they capture *all* the relative geometry of the pair:
#
# - the **dot product** $\langle u,v\rangle = |u||v|\cos\theta$ — **symmetric**, measuring *length and
#   angle* (how aligned the arrows are);
# - the **wedge product** $u\wedge v = |u||v|\sin\theta$ — **antisymmetric**, measuring *signed area*
#   (how much the arrows splay apart, and in which orientation).
#
# The slogan: **dot is the symmetric part, wedge is the antisymmetric part, of multiplying two
# vectors.** Cosine and sine; alignment and spread. And because $\cos^2+\sin^2=1$, the two satisfy a
# Pythagorean identity that lets you reconstruct the whole relative configuration of two arrows from
# just these numbers.
#
# We stay coordinate-free, build both products from their geometric meaning, and — for the first time
# — step into **3D**, where the wedge of two arrows becomes an oriented *area element* (a bivector),
# the thing that will give us volume and the determinant in notebook 4. Figures are k3d; the 3D ones
# are free to orbit (no top-down camera).

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
# ### Toolkit — now with filled parallelograms (2D and 3D)


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


def add_parallelogram(plot, o, u, v, color, opacity=0.35, outline=True):
    """Filled parallelogram with corner o and edges u, v (2D or 3D)."""
    o, u, v = np.asarray(o, float), np.asarray(u, float), np.asarray(v, float)
    corners = _to3(np.array([o, o + u, o + u + v, o + v]))
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    plot += k3d.mesh(corners, indices, color=color, opacity=opacity, side="double")
    if outline:
        loop = np.array([o, o + u, o + u + v, o + v, o])
        add_line(plot, loop, color, width=0.02)


# %% [markdown]
# ## 1. The dot product, properly
#
# Define $\langle u,v\rangle = |u|\,|v|\,\cos\theta$. Three structural facts make it the unique
# sensible "ruler":
#
# - **Symmetric**: $\langle u,v\rangle = \langle v,u\rangle$ (the angle does not care about order).
# - **Bilinear**: linear in each slot separately, $\langle au+bw, v\rangle = a\langle u,v\rangle + b\langle w,v\rangle$.
# - **Positive-definite**: $\langle v,v\rangle = |v|^2 \ge 0$, zero only for the zero arrow.
#
# From these alone come the headline results of plane trigonometry:
#
# - the **law of cosines** $|u-v|^2 = |u|^2 + |v|^2 - 2\langle u,v\rangle$ (just expand by bilinearity);
# - the **Cauchy–Schwarz inequality** $|\langle u,v\rangle| \le |u||v|$, with equality iff the arrows
#   are parallel — the statement that $|\cos\theta|\le 1$.


# %%
def dot(u, v):
    return np.sum(np.asarray(u, float) * np.asarray(v, float), axis=-1)


def norm(v):
    return np.sqrt(dot(v, v))


rng = np.random.default_rng(2)
u = np.array([2.0, 0.5])
v = np.array([0.6, 1.7])

# Bilinearity + symmetry, on random inputs.
sym = bilin = True
for _ in range(2000):
    p, q, r = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    a, b = rng.normal(), rng.normal()
    sym &= np.allclose(dot(p, q), dot(q, p))
    bilin &= np.allclose(dot(a * p + b * r, q), a * dot(p, q) + b * dot(r, q))
print("symmetric            :", sym)
print("bilinear (1st slot)  :", bilin)
print("positive-definite    :", dot(u, u) > 0 and np.isclose(dot(np.zeros(2), np.zeros(2)), 0))

# Law of cosines and Cauchy–Schwarz.
print("law of cosines       :", np.allclose(norm(u - v) ** 2, dot(u, u) + dot(v, v) - 2 * dot(u, v)))
print("Cauchy–Schwarz       :", abs(dot(u, v)) <= norm(u) * norm(v) + 1e-12)
print("  equality iff parallel:", np.allclose(abs(dot(u, 3.0 * u)), norm(u) * norm(3.0 * u)))


# %% [markdown]
# ## 2. The wedge product: signed area
#
# Now the antisymmetric partner. The **wedge** $u \wedge v$ is the **signed area** of the
# parallelogram spanned by $u$ and $v$:
#
# $$u \wedge v = |u|\,|v|\,\sin\theta.$$
#
# Its defining properties are the mirror image of the dot's:
#
# - **Antisymmetric**: $u\wedge v = -\,v\wedge u$. Swapping the arrows flips the orientation, hence the
#   sign. The magnitude is the area; the sign records whether $v$ is counter-clockwise from $u$ (+) or
#   clockwise (−).
# - **Bilinear**: linear in each slot (sliding one edge parallel to the other shears the parallelogram
#   without changing its area — base × height is unchanged).
# - **Alternating**: $u\wedge u = 0$. A degenerate parallelogram (two parallel edges) has no area.
#
# In the plane the wedge is a single signed number; concretely $u\wedge v = u_x v_y - u_y v_x$ (the
# 2D cross product we used for collinearity in notebook 1 — there, "lies on a line" meant "spans zero
# area").


# %%
def wedge2(u, v):
    """Signed area of the parallelogram spanned by u, v in the plane."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


# Antisymmetry, alternating, bilinearity, and the |u||v|sinθ identity.
alt = anti = bil = True
for _ in range(2000):
    p, q, r = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    a, b = rng.normal(), rng.normal()
    anti &= np.allclose(wedge2(p, q), -wedge2(q, p))
    alt &= np.allclose(wedge2(p, p), 0.0)
    bil &= np.allclose(wedge2(a * p + b * r, q), a * wedge2(p, q) + b * wedge2(r, q))
theta = np.arccos(np.clip(dot(u, v) / (norm(u) * norm(v)), -1, 1))
print("antisymmetric        :", anti)
print("alternating u∧u=0    :", alt)
print("bilinear             :", bil)
print("u∧v = |u||v|sinθ     :", np.allclose(wedge2(u, v), norm(u) * norm(v) * np.sin(theta)))

# The sign is orientation: u∧v > 0 means v is counter-clockwise from u.
p = new_plot(lim=3.0)
add_parallelogram(p, [0, 0], u, v, GREEN if wedge2(u, v) > 0 else RED)
add_vector(p, [0, 0], u, BLUE, "u")
add_vector(p, [0, 0], v, ORANGE, "v")
p.display()
print(f"signed area u∧v = {wedge2(u, v):+.3f}  (green = positive/counter-clockwise orientation)")

# %%
# Swapping the order flips the sign — same parallelogram, opposite orientation (drawn red).
p = new_plot(lim=3.0)
add_parallelogram(p, [0, 0], v, u, GREEN if wedge2(v, u) > 0 else RED)
add_vector(p, [0, 0], v, ORANGE, "v")
add_vector(p, [0, 0], u, BLUE, "u")
p.display()
print(f"signed area v∧u = {wedge2(v, u):+.3f}  (red = negative/clockwise — the orientation reversed)")


# %% [markdown]
# ## 3. Dot + wedge = the whole story
#
# Alignment is $\cos\theta$, spread is $\sin\theta$, and $\cos^2\theta + \sin^2\theta = 1$. Multiplying
# by $|u|^2|v|^2$ gives a **Pythagorean identity tying the two products together**:
#
# $$\langle u,v\rangle^2 + (u\wedge v)^2 = |u|^2\,|v|^2.$$
#
# So the dot and the wedge are genuinely the two orthogonal "components" of the pair $(u,v)$: knowing
# both lengths plus these two numbers pins down the relative geometry completely (up to a rigid
# motion). This is the 2D shadow of the **geometric product** $uv = \langle u,v\rangle + u\wedge v$,
# where a real "scalar" part and an "area" part are literally added — the idea geometric algebra is
# built on.

# %%
ok = True
for _ in range(5000):
    p_, q_ = rng.normal(size=2), rng.normal(size=2)
    ok &= np.allclose(dot(p_, q_) ** 2 + wedge2(p_, q_) ** 2, dot(p_, p_) * dot(q_, q_))
print("⟨u,v⟩² + (u∧v)² = |u|²|v|²  for 5000 random pairs:", ok)


# %% [markdown]
# ## 4. Areas of triangles and polygons — the shoelace formula *is* a sum of wedges
#
# Because the wedge is signed and bilinear, the area of any polygon is just half the sum of the
# wedges of consecutive vertex vectors — the **shoelace formula**, now revealed as "add up signed
# triangle areas, and the outside cancels":
#
# $$\text{Area} = \tfrac12\sum_i v_i \wedge v_{i+1}.$$
#
# The signed version even tells you the polygon's orientation (positive = counter-clockwise).


# %%
def polygon_area(verts):
    """Signed area via the shoelace = half-sum of consecutive wedges."""
    verts = np.asarray(verts, float)
    nxt = np.roll(verts, -1, axis=0)
    return 0.5 * np.sum(wedge2(verts, nxt))


# A pentagon, plus a triangle whose area we cross-check against ½|base×height|.
poly = np.array([[0.0, 0.0], [2.0, 0.3], [2.6, 1.8], [1.0, 2.6], [-0.5, 1.4]])
tri = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 2.0]])
print("triangle area (wedge)        :", polygon_area(tri))
print("triangle area (½ base·height):", 0.5 * 3.0 * 2.0)
print("pentagon signed area         :", polygon_area(poly), "(positive ⇒ counter-clockwise)")

p = new_plot(lim=3.2)
add_line(p, np.vstack([poly, poly[0]]), BLUE, width=0.03)
# Fan of signed triangles from the first vertex shows the cancellation visually.
for i in range(1, len(poly) - 1):
    add_parallelogram(p, poly[0], poly[i] - poly[0], poly[i + 1] - poly[0], GREEN, opacity=0.12, outline=False)
add_points(p, poly, ORANGE, size=0.12)
p.display()


# %% [markdown]
# ## 5. Into 3D: the wedge as an oriented area element
#
# In space, the parallelogram spanned by $u$ and $v$ still has an area, but now it also has an
# *orientation in 3D* — it tilts. The wedge $u\wedge v$ becomes a **bivector**: an oriented patch of
# plane whose magnitude is the area and whose attitude is the plane it lies in. The classical
# stand-in for that bivector is the **cross product** $u\times v$ — the vector perpendicular to the
# plane whose length equals the area and whose direction (right-hand rule) encodes the orientation:
#
# $$|u\times v| = |u|\,|v|\,\sin\theta = \text{area}, \qquad u\times v \perp u,\ v.$$
#
# The same Pythagorean identity holds in space: $|u\times v|^2 + \langle u,v\rangle^2 = |u|^2|v|^2$.
# Orbit the figure below — the green arrow is the area-vector standing normal to the blue/orange patch.


# %%
def cross3(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.array([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])


u3 = np.array([2.0, 0.4, 0.3])
v3 = np.array([0.5, 1.8, 0.2])
n3 = cross3(u3, v3)
area = np.linalg.norm(n3)
print("area |u×v|                 :", area)
print("u×v ⟂ u and ⟂ v            :", np.allclose(dot(n3, u3), 0), np.allclose(dot(n3, v3), 0))
print("|u×v|² + ⟨u,v⟩² = |u|²|v|² :", np.allclose(area**2 + dot(u3, v3) ** 2, dot(u3, u3) * dot(v3, v3)))

p = new_plot(lim=3.0, top_down=False, axes=False)
add_vector(p, [0, 0, 0], u3, BLUE, "u")
add_vector(p, [0, 0, 0], v3, ORANGE, "v")
add_parallelogram(p, [0, 0, 0], u3, v3, BLUE, opacity=0.3)
add_vector(p, [0, 0, 0], n3, GREEN, "u×v")  # area-vector normal to the patch
p.display()


# %% [markdown]
# ## 6. How operators act on area — the bridge to the determinant
#
# Here is the punchline that notebook 4 cashes in. Take *any* linear operator $T$ and feed a
# parallelogram through it. By bilinearity and antisymmetry of the wedge, the image area is the
# original area times a factor that **does not depend on which $u,v$ you chose**:
#
# $$T(u)\wedge T(v) = \big(\det T\big)\;(u\wedge v).$$
#
# That universal factor is the **determinant** — defined right here, coordinate-free, as *the number
# by which $T$ multiplies every signed area*. Let us see it is constant: pick a fixed $T$, throw
# random pairs at it, and watch the ratio settle on one value. Then read it off for the operators of
# notebook 2 — and the values match the determinant column we previewed there.


# %%
def apply_lin(A, v):
    """Apply the linear map with bookkeeping array A (2x2) to vectors / stacks of row-vectors."""
    return np.asarray(v, float) @ np.asarray(A, float).T


# A fixed but arbitrary linear map.
A = np.array([[1.3, -0.7], [0.4, 1.1]])
ratios = []
for _ in range(6):
    p_, q_ = rng.normal(size=2), rng.normal(size=2)
    ratios.append(wedge2(apply_lin(A, p_), apply_lin(A, q_)) / wedge2(p_, q_))
print("T(u)∧T(v) / (u∧v) for 6 random pairs:", np.round(ratios, 6))
print("→ all equal; this constant is det T =", round(ratios[0], 6))

# The operators of notebook 2, as area-scaling factors (compare nb 2's determinant column).
ops = {
    "rotation 36°": np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]]),
    "uniform scale 1.6": np.array([[1.6, 0], [0, 1.6]]),
    "shear k=0.8": np.array([[1, 0.8], [0, 1]]),
    "projection onto x": np.array([[1, 0], [0, 0]]),
    "reflection in x": np.array([[1, 0], [0, -1]]),
}
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
for name, M in ops.items():
    det = wedge2(apply_lin(M, e1), apply_lin(M, e2)) / wedge2(e1, e2)
    print(f"  {name:<20} area factor (det) = {det:+.3f}")


# %% [markdown]
# As promised in notebook 2: rotation and shear preserve area ($\det=+1$), uniform scaling multiplies
# it by $\lambda^2$ ($1.6^2=2.56$), projection crushes it to $0$, and reflection gives $-1$ — the
# minus sign is the orientation flip the wedge was built to detect.

# %% [markdown]
# ## Summary
#
# - The **dot product** is the *symmetric* bilinear pairing $|u||v|\cos\theta$: lengths, angles,
#   perpendicularity, the law of cosines, Cauchy–Schwarz.
# - The **wedge product** is the *antisymmetric* bilinear pairing $|u||v|\sin\theta$: **signed area**,
#   antisymmetric, alternating ($u\wedge u=0$). Its sign is orientation.
# - They are two halves of one product, locked together by $\langle u,v\rangle^2 + (u\wedge v)^2 =
#   |u|^2|v|^2$ — the geometric-product identity in miniature.
# - **Area of any polygon** = half-sum of consecutive wedges (the shoelace formula).
# - In **3D** the wedge is an oriented area element (bivector), classically the **cross product**:
#   length = area, direction ⟂ the plane.
# - **Operators scale every signed area by one universal factor** $T(u)\wedge T(v)=(\det T)(u\wedge v)$
#   — the coordinate-free **determinant**, taken up next.
#
# **Next (notebook 4):** the top-degree wedge as the **volume element**, the **determinant** as the
# volume-scaling factor (now in 2D *and* 3D), and a **coordinate-free trace** as the first-order rate
# at which an operator changes volume.
