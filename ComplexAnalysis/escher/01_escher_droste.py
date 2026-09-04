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
# # The logarithm of an image — Escher's *Print Gallery*
#
# After 3Blue1Brown's *[How (and why) to take a logarithm of an image](https://youtu.be/ldxFjLJ3rVY)*
# and the paper behind it: **B. de Smit and H. W. Lenstra Jr., "The Mathematical Structure of Escher's
# *Print Gallery*"**, *Notices of the AMS* **50** (2003) 446–451 — PDF, extracted figures and the video
# all sit next to this notebook in `ComplexAnalysis/escher/`.
#
# ## The whole story in four symbols
#
# There are two pictures and one map between them.
#
# * The **straight world**: an ordinary, undistorted drawing $f$ that contains a copy of itself shrunk
#   by 256. Escher drew it as four studies, one per corner, each showing a portion of the previous one
#   blown up by 4. Mathematically $$f(256\,z) = f(z).$$
# * The **curved world**: the lithograph $g$, ideally periodic under a *complex* multiplier
#   $$
#   g(\gamma\,w) = g(w),\qquad \gamma\in\mathbb{C}^*$$ — rotate-and-shrink, and the picture is
#   unchanged.
# * The dictionary between them is Escher's grid, which he wanted **conformal** (in Bruno Ernst's
#   report of his own words: so the original squares could "better retain their square appearance").
#   A conformal isomorphism $\mathbb{C}^*/\langle\gamma\rangle \to \mathbb{C}^*/\langle 256\rangle$ is
#   forced to be a power map: $$h(w) = w^{\alpha},\qquad g = f\circ h.$$
# * Chasing the loop $A\to B\to C\to D\to A$ around Escher's grid (paper Figures 4 and 7) pins
#   $\alpha$ down completely:
#   $$\boxed{\;\alpha=\frac{2\pi i+\log 256}{2\pi i}\;},\qquad \log\gamma=\frac{\log 256}{\alpha}.$$
#
# **Everything below follows from one line of code** — take the logarithm, multiply by $\alpha$,
# exponentiate back:
#
# > `z = np.exp(ALPHA * np.log(w))`
#
# ### A note on the pictures used here
#
# Escher's lithograph is under copyright (Cordon Art B.V.), so this notebook does not reproduce it.
# It builds instead a synthetic "print gallery" with exactly the same symmetry structure — which is
# better for seeing the geometry anyway, because you know what the shapes started as.

# %%
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import clmmathtools.viz as mv

mv.set_backend("mpl")

# %% [markdown]
# ## 1. The constants
#
# The straight picture repeats under multiplication by 256. Escher's *four* studies mean it in fact
# repeats under multiplication by $4$ **and** by $i$ — and $4^4 = 256$, $i^4 = 1$ (paper, Figure 10
# caption). We keep both.

# %%
S_BIG = 256.0  # the straight picture's multiplicative period
S = 4.0  # ... and its finer period: four studies, S**4 == S_BIG

LOG_S_BIG = np.log(S_BIG)
ALPHA = (2j * np.pi + LOG_S_BIG) / (2j * np.pi)
BETA = 1.0 / ALPHA
LOG_GAMMA = LOG_S_BIG / ALPHA
GAMMA = np.exp(LOG_GAMMA)

print(f"alpha     = {ALPHA.real:.10f} {ALPHA.imag:+.10f} i")
print(f"log gamma = {LOG_GAMMA.real:.10f} {LOG_GAMMA.imag:+.10f} i")
print(f"|gamma|   = {abs(GAMMA):.10f}")
print(f"arg gamma = {np.degrees(np.angle(GAMMA)):.10f}° counter-clockwise")
print(
    f"  ... so 1/gamma is a clockwise turn of {-np.degrees(np.angle(GAMMA)) % 360:.10f}°"
)

# the paper's headline numbers, to ten digits
assert np.isclose(abs(GAMMA), 22.5836845286, atol=1e-9)
assert np.isclose(np.degrees(np.angle(GAMMA)), 157.6255960832, atol=1e-9)
print(
    "\n✓ matches the paper: shrink by 22.5836845286…, rotate clockwise 157.6255960832…°"
)

# %% [markdown]
# ### Why that $\alpha$ forces that $\gamma$ — one line of algebra
#
# $h(w)=w^\alpha$, so
# $$h(\gamma w)=(\gamma w)^\alpha=\gamma^\alpha\,h(w),\qquad
#   \gamma^\alpha=e^{\alpha\log\gamma}=e^{\alpha\cdot(\log 256)/\alpha}=256,$$
# hence $g(\gamma w)=f\bigl(256\,h(w)\bigr)=f\bigl(h(w)\bigr)=g(w)$. **The complex period of the
# lithograph is nothing but the real period 256 of the studies, read through $h$.**

# %%
assert np.isclose(GAMMA**ALPHA, S_BIG)
assert np.isclose(np.exp(LOG_GAMMA / 4) ** ALPHA, S)  # γ^¼ pulls back to 4
assert np.isclose(1j**ALPHA, 4j)  # a quarter turn pulls back to ×4i
print("gamma**alpha    =", np.round(GAMMA**ALPHA, 9))
print("(gamma**¼)**alpha =", np.round(np.exp(LOG_GAMMA / 4) ** ALPHA, 9), " (= 4)")
print("i**alpha        =", np.round(1j**ALPHA, 9), " (= 4i)")

# %% [markdown]
# ## 2. The straight world: an ordinary drawing that contains itself
#
# First we need $f$: a perfectly **undistorted** picture — straight lines, square frames — that
# satisfies $f(4z)=f(z)$ and $f(iz)=f(z)$.
#
# The natural fundamental domain for "multiply by 4" is a **square annulus**. Write
# $L(z)=\max(\lvert\operatorname{Re}z\rvert,\lvert\operatorname{Im}z\rvert)$; then $L(4z)=4L(z)$ and
# $L(iz)=L(z)$, so the ring $1\le L(z)<4$ is a fundamental domain for $\langle 4\rangle$, and $\times i$
# rotates it onto itself. Draw one gallery wall on that ring, hang four framed prints on it, and
# repeat it at every scale $4^k$.
#
# The one constraint is **seamlessness**: nothing may touch $L=1$ or $L=4$, or the picture will not
# match up across the fold. A plain margin of wall at both boundaries is what buys that — and a plain
# margin of wall is, conveniently, what a gallery looks like.

# %%
MOTIF_EXTENT = 4.0  # the raster covers [-4, 4]^2 = the outer square of the ring
WALL = "#161a22"


def draw_gallery(px=1000, frame="#e0b352", mat="#f2ede1"):
    """Raster the fundamental square annulus 1 ≤ L(z) < 4 over [-4,4]², one quadrant's worth.

    Elements are drawn **once** on a transparent canvas and then composited with their three
    quarter-turn rotations, so f(iz) = f(z) holds exactly (bit-for-bit) rather than up to whatever
    the rasterizer does to four separately-drawn copies.
    """
    fig = Figure(figsize=(px / 100, px / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-MOTIF_EXTENT, MOTIF_EXTENT)
    ax.set_ylim(-MOTIF_EXTENT, MOTIF_EXTENT)
    ax.set_facecolor("none")
    ax.axis("off")

    def print_on_wall(cx, cy, half, label):
        """A framed print: gilt frame, cream mat, an abstract quayside inside."""
        ax.add_patch(
            Rectangle((cx - half, cy - half), 2 * half, 2 * half, fc=frame, ec="none")
        )
        m = half * 0.86
        ax.add_patch(Rectangle((cx - m, cy - m), 2 * m, 2 * m, fc=mat, ec="none"))
        s = half * 0.74  # the picture inside the mat
        ax.add_patch(
            Rectangle((cx - s, cy - s), 2 * s, s * 0.85, fc="#3f7fa6", ec="none")
        )  # water
        for frac, hgt, col in [
            (0.05, 0.60, "#c25b4a"),
            (0.38, 0.95, "#7fa85f"),
            (0.70, 0.45, "#9a6fb0"),
        ]:
            ax.add_patch(
                Rectangle(
                    (cx - s + frac * 2 * s, cy - s + s * 0.85),
                    2 * s * 0.24,
                    s * hgt,
                    fc=col,
                    ec="none",
                )
            )  # quayside buildings
        ax.text(
            cx + s * 0.72,
            cy - s * 0.78,
            label,
            ha="right",
            va="bottom",
            fontsize=half * 22,
            color="#22262e",
            fontweight="bold",
        )  # asymmetric, so rotations are legible

    # one print centred on the +x axis, one on the diagonal — both strictly inside 1 < L < 4
    print_on_wall(2.45, 0.0, 0.95, "R")  # L ∈ [1.50, 3.40]
    print_on_wall(2.60, 2.60, 1.05, "R")  # L ∈ [1.55, 3.65]

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(np.float64) / 255.0
    rgba = rgba[::-1]  # row 0 = y_min, matching a raster drawn with origin="lower"

    # composite the quadrant over the wall, four quarter-turns of it, exactly
    out = np.tile(np.array(mv.palette.rgb01(WALL)), (*rgba.shape[:2], 1))
    for k in range(4):
        layer = np.rot90(rgba, k, axes=(0, 1))
        a = layer[..., 3:4]
        out = layer[..., :3] * a + out * (1 - a)
    return out


GALLERY = draw_gallery()
print("motif raster:", GALLERY.shape, f"over [-{MOTIF_EXTENT}, {MOTIF_EXTENT}]²")

# %% [markdown]
# `f_straight` is the whole self-similar picture: fold $z$ into the ring by dividing out the right
# power of 4, then look it up. Note there is no rotational folding — the raster is already exactly
# quarter-turn symmetric, which is what makes $f(iz)=f(z)$ true to machine precision.

# %%
BG = np.array(mv.palette.rgb01(mv.BG))
LOG_S = np.log(S)


def f_straight(z):
    """The undistorted, self-similar drawing: f(4z) = f(iz) = f(z). Returns (..., 3) RGB."""
    z = np.asarray(z, dtype=complex)
    L = np.maximum(np.abs(z.real), np.abs(z.imag))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.floor(np.log(L) / LOG_S)  # which ring z sits in
        zf = z * np.power(S, -k)  # ... folded back to 1 ≤ L < 4
    H, W = GALLERY.shape[:2]
    x = (zf.real + MOTIF_EXTENT) / (2 * MOTIF_EXTENT) * W - 0.5
    y = (zf.imag + MOTIF_EXTENT) / (2 * MOTIF_EXTENT) * H - 0.5
    x = np.clip(np.nan_to_num(x, nan=0.0), 0, W - 1.001)
    y = np.clip(np.nan_to_num(y, nan=0.0), 0, H - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = GALLERY[y0, x0] * (1 - fx) + GALLERY[y0, x0 + 1] * fx
    bot = GALLERY[y0 + 1, x0] * (1 - fx) + GALLERY[y0 + 1, x0 + 1] * fx
    rgb = top * (1 - fy) + bot * fy
    rgb[~np.isfinite(k)] = BG  # z = 0, or ±inf
    return rgb


def render(zmap, extent=1.0, res=560, ss=3, hole=None, center=0j):
    """Rasterize the picture seen through `zmap`: screen point w ↦ straight-world point z.

    Supersampled `ss`× then box-averaged. Towards the origin the map crams unboundedly many periods
    into one pixel; without oversampling that reads as moiré rather than as detail. `hole` blanks a
    small central disk — the paper's hole, "too small to notice".
    """
    n = res * ss
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs)
    w = center + X + 1j * Y
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = f_straight(zmap(w))
    if hole is not None:
        rgb[np.abs(w - center) < hole] = BG
    return rgb.reshape(res, ss, res, ss, 3).mean(axis=(1, 3))


# %% [markdown]
# Here is $f$ — Escher's studies, idealized. Straight lines, square frames, and the same wall
# reappearing four times smaller in every direction as you approach the centre.

# %%
IMG_STRAIGHT = render(lambda w: w, extent=6.0, res=640, hole=0.02)
mv.Plane(extent=1, grid=False, axes=False).raster(IMG_STRAIGHT).display(format="png")

# %% [markdown]
# The three declared symmetries, checked on actual pixels rather than asserted in prose:

# %%
probe = np.array([2.31 + 1.17j, -1.22 + 3.44j, 0.55 - 2.63j, -3.51 - 0.28j])
for name, mult in [
    ("f(4z) = f(z)", 4.0),
    ("f(iz) = f(z)", 1j),
    ("f(256z) = f(z)", 256.0),
]:
    assert np.allclose(f_straight(probe * mult), f_straight(probe), atol=1e-12)
    print(f"✓ {name}")

# %% [markdown]
# ## 3. Taking the logarithm (paper, Figure 14)
#
# Set $\xi=\log z$. Multiplication becomes translation, so the two symmetries become a **lattice**:
#
# | straight-world symmetry | log-space symmetry |
# |---|---|
# | $f(4z)=f(z)$ | $\xi \mapsto \xi + \log 4$ |
# | $f(iz)=f(z)$ | $\xi \mapsto \xi + i\pi/2$ |
# | $f(256z)=f(z)$ | $\xi \mapsto \xi + \log 256$ |
#
# The picture becomes **doubly periodic on $\mathbb{C}$** — a wallpaper pattern, one period
# $\log 256$ wide and $2\pi i$ tall. This is the object de Smit and Lenstra handed to Jacqueline
# Hofstra to paint the grayscale into, precisely because it is the only place where the picture has no
# preferred centre and no accumulating detail.
#
# Note the frames come out *bent*: `exp` is conformal but it is not affine, and straightness is not
# something it preserves.


# %%
def render_log(u0, u1, v0, v1, res=560, ss=2):
    """Rasterize the picture in log space: ξ = u + iv ↦ f(exp(ξ))."""
    nv = res
    nu = max(2, int(round(res * (u1 - u0) / (v1 - v0))))
    us = np.linspace(u0, u1, nu * ss)
    vs = np.linspace(v0, v1, nv * ss)
    U, V = np.meshgrid(us, vs)
    rgb = f_straight(np.exp(U + 1j * V))
    return rgb.reshape(nv, ss, nu, ss, 3).mean(axis=(1, 3))


# a Plane's view is always the square [-E, E]², so centre the tile in it: log 256 < 2π, so a view
# of extent π holds one full period with a sliver of margin left and right.
U0, U1 = -LOG_S_BIG / 2, LOG_S_BIG / 2
LOG_TILE = render_log(U0, U1, -np.pi, np.pi, res=520)

# outline one fundamental cell, log 4 wide by π/2 tall (a closed curve, not a Polygon: a Polygon's
# alpha applies to its edge as well, so an unfilled one cannot be drawn that way)
c0 = np.array([U0, -np.pi])
cell = np.array([c0, c0 + [LOG_S, 0], c0 + [LOG_S, np.pi / 2], c0 + [0, np.pi / 2], c0])
(
    mv.Plane(extent=np.pi, grid=False, axes=False)
    .raster(LOG_TILE, extent=(U0, U1, -np.pi, np.pi))
    .curve(cell, color=mv.ORANGE, width=2.5)
    .display(format="png")
)
print(
    f"one full period: {LOG_S_BIG:.4f} (= log 256) wide × {2 * np.pi:.4f} (= 2π) tall"
)
print(
    f"    finer cell:  {LOG_S:.4f} (= log 4)   wide × {np.pi / 2:.4f} (= π/2) tall  (outlined)"
)

# %% [markdown]
# ### The lattices (paper, Figure 10)
#
# $L_{256}=\mathbb{Z}\,2\pi i+\mathbb{Z}\log 256$ is the fundamental group of
# $\mathbb{C}^*/\langle 256\rangle$; $L_\gamma=\alpha^{-1}L_{256}$ is that of
# $\mathbb{C}^*/\langle\gamma\rangle$. The map $h$ lifts to **multiplication by the scalar $\alpha$**,
# and multiplying a lattice by a complex scalar is just a rotation and a scaling — which is the entire
# geometric content of the theorem.


# %%
def lattice_points(a, b, r=3):
    ks = np.arange(-r, r + 1)
    M, N = np.meshgrid(ks, ks)
    z = M * a + N * b
    return np.c_[z.real.ravel(), z.imag.ravel()]


for label, (a, b), col in [
    ("L_256  (upright)", (complex(LOG_S), 1j * np.pi / 2), mv.BLUE),
    ("L_gamma = α⁻¹·L_256", (BETA * LOG_S, 1j * BETA * np.pi / 2), mv.ORANGE),
]:
    print(f"{label}:  basis {a:.4f}, {b:.4f}")
    (
        mv.Plane(extent=3.2, grid=False)
        .points(lattice_points(a, b), color=col, size=22)
        .parallelogram(
            [0, 0], [a.real, a.imag], [b.real, b.imag], facecolor=col, alpha=0.25
        )
        .vector([a.real, a.imag], color=col, label="log 4 / α")
        .vector([b.real, b.imag], color=col, label="iπ/2 / α")
        .display(format="png")
    )
print(
    f"β = 1/α rotates by {np.degrees(np.angle(BETA)):.2f}° and scales by {abs(BETA):.4f}"
)

# %% [markdown]
# ## 4. The map — and the lithograph
#
# $g = f\circ h$ with $h(w)=w^{\alpha}=\exp(\alpha\log w)$. That is the entire transformation:


# %%
def h(w):
    """Escher's dictionary: curved world → straight world, w ↦ w^α. Pointwise, principal branch."""
    return np.exp(ALPHA * np.log(w))


def _continuous_log(z):
    """log z with the branch tracked continuously **along a sampled path** (NaNs passed through).

    `np.log` takes the principal branch pointwise, which tears any path crossing the negative real
    axis. Transporting a *curve* — as every argument in the paper does — needs the branch to follow
    the curve instead, which is what `np.unwrap` supplies.
    """
    z = np.asarray(z, dtype=complex)
    ang = np.full(z.shape, np.nan)
    ok = np.isfinite(z)
    if ok.any():
        ang[ok] = np.unwrap(np.angle(z[ok]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.abs(z)) + 1j * ang


def to_straight(w):
    """Push a curved-world *path* forward: z = w^α, branch tracked along the path."""
    return np.exp(ALPHA * _continuous_log(w))


def to_curved(z):
    """Pull a straight-world *path* back: w = z^(1/α) = exp(β log z), branch tracked."""
    return np.exp(BETA * _continuous_log(z))


def clip_lines(lines, extent, rmin=0.0):
    """Complex polylines → (N,2) arrays, NaN-ing whatever leaves the view so the line breaks there.

    `rmin` additionally punches out a disk at the centre, which keeps a grid that accumulates at the
    origin from turning into a solid blob of ink.
    """
    out = []
    for z in lines:
        z = np.asarray(z, dtype=complex)
        keep = (
            (np.abs(z.real) <= extent)
            & (np.abs(z.imag) <= extent)
            & (np.abs(z) >= rmin)
        )
        z = np.where(keep, z, np.nan)
        if np.isfinite(z).any():
            out.append(np.c_[z.real, z.imag])
    return out


IMG_ESCHER = render(h, extent=1.0, res=720, hole=0.003)
mv.Plane(extent=1, grid=False, axes=False).raster(IMG_ESCHER).display(format="png")

# %% [markdown]
# ## 5. The grid — where Escher's headaches were

# %% [markdown]
# ### 5a. Escher's two false starts (paper, Figures 2 and 3)
#
# Before arriving at the grid of Figure 4, Escher tried a "cyclic expansion … without beginning or
# end" twice. In Bruno Ernst's account he first tried straight lines, then "intuitively adopted"
# curved ones — because that way, in the phrase the whole paper turns on, the original small squares
# could better retain their square appearance.
#
# Both attempts share one skeleton, and we can build it exactly. A cyclic expansion by a factor $E$
# means one circuit of the centre multiplies scale by $E$ — in log space, the vector
# $$\text{circuit} = \log E + 2\pi i.$$
# Cut it into $n$ equal steps $a$, take $b = i\,a$ (same length, at right angles), and
# $\mathbb{Z}a+\mathbb{Z}b$ is a **square** lattice; since `exp` is conformal, its image has square
# cells too. Those vertices are common to both figures. The attempts differ *only in how neighbouring
# vertices are joined*:
#
# * **Figure 2** joins them by straight chords drawn in the plane;
# * **Figure 3** joins them along the conformal curves — the images under `exp` of the straight
#   log-space lines, which are logarithmic spirals.

# %%
N_CELLS = 12  # cells per circuit; Escher's sketches are coarse, and coarse is where it goes wrong


def cyclic_expansion(expansion=4.0, n=N_CELLS, jr=(-17, 17), kr=(-1, 2)):
    """Log-space vertices of a cyclic expansion: one circuit multiplies scale by `expansion`."""
    a = (np.log(expansion) + 2j * np.pi) / n
    b = 1j * a  # same length, at right angles ⇒ square cells
    js = np.arange(jr[0], jr[1] + 1)[:, None]
    ks = np.arange(kr[0], kr[1] + 1)[None, :]
    return js * a + ks * b, a, b


CYC_ZETA, AVEC, BVEC = cyclic_expansion()
CYC_VERT = np.exp(CYC_ZETA)  # the vertices both figures share

# one circuit of n steps really is a blow-up by 4 — and exp(2πi) = 1 is what makes it close up
assert np.allclose(np.exp(CYC_ZETA + N_CELLS * AVEC), 4.0 * CYC_VERT)
print(
    f"{N_CELLS} steps around  ⇒  scale ×{np.exp(N_CELLS * AVEC).real:.6f}  (exactly 4) ✓"
)


def chord_family(vertices):
    """Figure 2: polylines through the vertices — i.e. straight segments, drawn in the plane."""
    return list(vertices) + list(vertices.T)


def arc_family(zeta, samples=40):
    """Figure 3: exp of the straight log-space lines through the same vertices — log spirals."""
    out = []
    for grid in (zeta, zeta.T):
        for row in grid:
            idx = np.arange(len(row))
            t = np.linspace(0, len(row) - 1, (len(row) - 1) * samples + 1)
            out.append(
                np.exp(np.interp(t, idx, row.real) + 1j * np.interp(t, idx, row.imag))
            )
    return out


E23, R23 = 3.0, 0.10
print("\nFigure 2 — a cyclic expansion drawn with straight lines")
p = mv.Plane(extent=E23, grid=False, axes=False)
for pts in clip_lines(chord_family(CYC_VERT), E23, R23):
    p.curve(pts, color=mv.RED, width=1.5)
p.display(format="png")

print("Figure 3 — the same vertices, joined conformally")
p = mv.Plane(extent=E23, grid=False, axes=False)
for pts in clip_lines(arc_family(CYC_ZETA), E23, R23):
    p.curve(pts, color=mv.GREEN, width=1.5)
p.display(format="png")

# %% [markdown]
# ### Measuring "square appearance"
#
# The complaint is quantifiable: take each cell's four corners and see how far its interior angles
# stray from 90°.
#
# For the conformal version there is nothing to measure — the two families cross at
# $\arg(b/a)=\arg(i)=90°$ *exactly*, at every vertex, and `exp` preserves that. For the straight
# version the answer is a single number, the same for every cell in the picture, because the grid is
# scale-invariant: **Escher's first attempt is uniformly wrong, everywhere at once.**


# %%
def corner_errors(vertices):
    """Max deviation from 90° of each straight-sided cell's interior angles, in degrees."""
    out = []
    for j in range(vertices.shape[0] - 1):
        for k in range(vertices.shape[1] - 1):
            q = [
                vertices[j, k],
                vertices[j + 1, k],
                vertices[j + 1, k + 1],
                vertices[j, k + 1],
            ]
            ang = [
                np.degrees(
                    abs(np.angle((q[(i + 1) % 4] - q[i]) / (q[(i - 1) % 4] - q[i])))
                )
                for i in range(4)
            ]
            out.append(np.abs(np.array(ang) - 90).max())
    return np.array(out)


err = corner_errors(CYC_VERT)
cross = np.degrees(abs(np.angle(BVEC / AVEC)))
print(
    f"conformal (Figure 3): the families cross at {cross:.6f}° — exact, at every vertex"
)
print(f"straight  (Figure 2): every cell is off by {err.mean():.2f}°")
print(
    f"                      spread over all {err.size} cells: {np.ptp(err):.1e} — all alike"
)
assert np.isclose(abs(np.angle(BVEC / AVEC)), np.pi / 2)
assert np.ptp(err) < 1e-9 < 1.0 < err.mean()

# the straight version recovers squareness only by making the cells too small to draw a gallery in
print("\n cells/turn   corner error of the straight cells")
for nc in (8, 12, 16, 24, 48, 96):
    z, _, _ = cyclic_expansion(n=nc, jr=(-2 * nc, 2 * nc))
    print(f"   {nc:3d}          {corner_errors(np.exp(z)).mean():6.2f}°")
print(
    "\n→ halving the error means doubling the cell count. Escher needed cells big enough to draw a"
)
print("  gallery inside, so the straight construction was never going to work.")

# %% [markdown]
# ### 5b. The mechanism: `exp` of a lattice, tilted
#
# Take the Cartesian grid aligned with $L_{256}$ in log space. Escher's proportions are four rings per
# factor of 4 and sixteen cells around, giving a log-space cell of
# $\Delta u=\tfrac{\log 4}{4}$ by $\Delta v=\tfrac{2\pi}{16}$ — nearly square, which is exactly the
# "squares retaining their square appearance" he was after.
#
# `exp` of that upright grid is the log-polar grid on the studies. `exp` of the *same* grid multiplied
# by $\beta=1/\alpha$ — i.e. rotated 41.4° and scaled 0.75 — is Escher's curved grid, the paper's
# **Figure 11**. Same lattice; one complex multiplication apart.

# %%
DU, DV = LOG_S / 4, 2 * np.pi / 16
print(f"log-space cell: {DU:.4f} × {DV:.4f}  (aspect {DU / DV:.3f} — near-square ✓)")


def lattice_lines(a, b, box, samples=1200):
    """Grid lines of the lattice ℤa + ℤb clipped to box = (umin, umax, vmin, vmax) ⊂ ℂ.

    Returns complex arrays, one per line, with out-of-box points set to NaN so the polyline simply
    breaks there rather than streaking across the figure.
    """
    umin, umax, vmin, vmax = box
    A = np.array([[a.real, b.real], [a.imag, b.imag]])
    mn = np.linalg.solve(
        A, np.array([[umin, umax, umax, umin], [vmin, vmin, vmax, vmax]])
    )
    m0, m1 = int(np.floor(mn[0].min())) - 1, int(np.ceil(mn[0].max())) + 1
    n0, n1 = int(np.floor(mn[1].min())) - 1, int(np.ceil(mn[1].max())) + 1
    lines = [m * a + np.linspace(n0, n1, samples) * b for m in range(m0, m1 + 1)]
    lines += [n * b + np.linspace(m0, m1, samples) * a for n in range(n0, n1 + 1)]
    return [
        np.where(
            (z.real >= umin) & (z.real <= umax) & (z.imag >= vmin) & (z.imag <= vmax),
            z,
            np.nan,
        )
        for z in lines
    ]


def exp_lines(lines, extent):
    """exp() each log-space polyline, then clip to the view."""
    return clip_lines([np.exp(np.asarray(z, dtype=complex)) for z in lines], extent)


def grid_at(extent, basis, inner=0.012):
    """The grid of §5a rendered for a square view of half-width `extent`.

    The log-space box spans Im ζ ∈ [-π, π], one full strip — and `exp` maps that strip onto all of
    ℂ*, so one strip is the whole picture, not a slice of it.
    """
    a, b = basis
    box = (np.log(inner * extent), np.log(1.6 * extent), -np.pi, np.pi)
    return exp_lines(lattice_lines(a, b, box), extent)


UPRIGHT = (complex(DU), 1j * DV)  # the lattice on the studies
TILTED = (BETA * DU, 1j * BETA * DV)  # ... times β = 1/α: Escher's grid

straight_grid = grid_at(1.0, UPRIGHT)
escher_grid = grid_at(1.0, TILTED)

p = mv.Plane(extent=1, grid=False, axes=False)
for pts in straight_grid:
    p.curve(pts, color=mv.BLUE, width=1.0, alpha=0.9)
p.display(format="png")

p = mv.Plane(extent=1, grid=False, axes=False)
for pts in escher_grid:
    p.curve(pts, color=mv.ORANGE, width=1.0, alpha=0.9)
p.display(format="png")

# %% [markdown]
# ### 5c. Escher's grid proper: the square grid pulled back (paper, Figures 4 and 6)
#
# What Escher actually fitted his studies onto was a **square** grid — the quadtree grid on the
# straight drawing, doubling as it moves outward — pulled back through $h$. Pulling a curve back means
# $w = \exp\bigl(\log z / \alpha\bigr)$, and the branch of $\log$ has to be chosen *continuously along
# each line*, which is what `np.unwrap` is doing below.
#
# Branch choices do not change the picture: shifting $\log z$ by $2\pi i$ multiplies $w$ by
# $\exp(2\pi i/\alpha)$, which lies in the symmetry group of the grid. A different branch simply drops
# the segment onto a different copy of itself.


# %%
def square_grid_lines(kmin=-4, kmax=3, cells=4, samples=1400):
    """The self-similar square grid on the straight drawing: in the ring 2^k ≤ L(z) < 2^(k+1), a
    square grid of side 2^k / cells. Doubling the cell each ring is what keeps it self-similar."""
    lines = []
    for k in range(kmin, kmax + 1):
        a, s = 2.0**k, 2.0**k / cells
        t = np.linspace(-2 * a, 2 * a, samples)
        for j in range(-2 * cells, 2 * cells + 1):
            c = j * s
            for z in (t + 1j * c, c + 1j * t):
                L = np.maximum(np.abs(z.real), np.abs(z.imag))
                lines.append(np.where((L >= a) & (L < 2 * a), z, np.nan))
    return lines


# a branch shift of 2πi in log z multiplies w by exp(2πi/α) — which is exactly 1/γ, a symmetry of
# the grid. So `np.unwrap` picking one branch per line costs nothing: replicating each pulled-back
# line by γ**n covers every branch and fills the view.
assert np.isclose(np.exp(2j * np.pi * BETA), 1 / GAMMA)
print(
    f"exp(2πi/α) = {np.exp(2j * np.pi * BETA):.6f} = 1/γ ✓  (a branch shift is a γ-symmetry)"
)


def pull_back(lines, extent, copies=range(-1, 3)):
    """h⁻¹ applied to straight-world polylines: w = exp(log z / α), branch unwrapped along the line."""
    out = []
    for z in lines:
        if np.isfinite(z).sum() < 2:
            continue
        w0 = to_curved(z)
        for n in copies:
            w = w0 * GAMMA**n
            w = np.where(
                (np.abs(w.real) <= extent) & (np.abs(w.imag) <= extent), w, np.nan
            )
            if np.isfinite(w).any():
                out.append(np.c_[w.real, w.imag])
    return out


SQ = square_grid_lines(cells=3)
p = mv.Plane(extent=6, grid=False, axes=False).raster(IMG_STRAIGHT, alpha=0.9)
for pts in clip_lines(SQ, 6.0):
    p.curve(pts, color=mv.BLUE, width=0.9, alpha=0.75)
p.display(format="png")

p = mv.Plane(extent=1, grid=False, axes=False).raster(IMG_ESCHER, alpha=0.9)
for pts in pull_back(SQ, 1.0):
    p.curve(pts, color=mv.ORANGE, width=0.9, alpha=0.75)
p.display(format="png")

# %% [markdown]
# ## 6. Loop transport — how $\gamma$ was *measured* (paper, Figures 7, 8, 9)
#
# Sections 1–5 **derived** $\gamma$ from $\alpha$. The paper reached it the other way round first, by
# walking loops on the grid and watching where they came back — and that is the argument that makes
# the model believable in the first place. It needs exactly one new tool: a way to ask whether a path
# encloses the origin.
#
# ### The winding number
#
# $$n(\Gamma)=\frac{1}{2\pi i}\oint_\Gamma\frac{dz}{z}=\frac{1}{2\pi}\Bigl[\arg z\Bigr]_\Gamma$$
#
# Since $d(\log z) = dz/z$ and $\log z = \log\lvert z\rvert + i\arg z$, and $\log\lvert z\rvert$
# returns to its starting value around a closed loop, the integral collapses to the **total turning of
# $\arg z$** — which is exactly what `np.unwrap` accumulates. Discretizing it that way is not an
# approximation; it is exact for a polyline.
#
# Two things can break it, and both are worth asserting rather than hoping for:
#
# * the path must stay clear of $0$, where $\arg$ is undefined and the integrand has its pole;
# * consecutive samples must turn by **less than $\pi$**, or `unwrap` cannot tell a small positive
#   turn from a large negative one and will silently miscount.
#
# That second guard is the price of using the integral rather than a bounding-box test — and it is
# what buys the generality: this works for any sampled path, not just axis-aligned squares.


# %%
def winding_number(path, *, closed=True, r_tol=1e-9, step_tol=0.99 * np.pi):
    """(1/2πi)∮dz/z about the origin, as the total turning of arg z along a sampled path.

    Raises rather than returning a quietly wrong integer: near the pole at 0 the quantity is not
    defined, and under-sampling makes the branch tracking ambiguous. With ``closed=True`` the result
    must additionally come out an integer, which is a free consistency check on the sampling.
    """
    z = np.asarray(path, dtype=complex)
    if np.abs(z).min() < r_tol:
        raise ValueError(
            "path passes through the origin — the winding number is undefined there"
        )
    step = np.abs(np.diff(np.angle(z)))
    step = np.minimum(
        step, 2 * np.pi - step
    )  # the true turn, whichever way round the cut
    if step.max() >= step_tol:
        raise ValueError(
            f"path under-sampled: one step turns by {step.max():.3f} rad ≈ π — refine before unwrapping"
        )
    turn = np.unwrap(np.angle(z))
    n = (turn[-1] - turn[0]) / (2 * np.pi)
    if not closed:
        return n
    if not np.isclose(n, round(n), atol=1e-6):
        raise ValueError(
            f"closed path, but the winding number came out {n:.6f}, not an integer"
        )
    return int(
        round(n)
    )  # a closed path's winding number is an integer — hand back an integer


def walk(start, legs, samples=400):
    """Sample the polyline that starts at `start` and follows each complex displacement in `legs`."""
    z = complex(start)
    pts = [np.array([z])]
    for d in legs:
        pts.append(z + np.linspace(0, 1, samples)[1:] * d)
        z += d
    return np.concatenate(pts)


def fit_extent(*paths, margin=1.12):
    """The half-width of the smallest origin-centred square view holding every path."""
    reach = [
        np.maximum(np.abs(np.asarray(q).real), np.abs(np.asarray(q).imag))
        for q in paths
    ]
    return float(np.nanmax(np.concatenate(reach)) * margin)


def as_xy(z):
    z = np.asarray(z, dtype=complex)
    return np.c_[z.real, z.imag]


# both guards, on paths that genuinely break the computation (a 4-corner square does not: its
# steps only ever turn by ~125°, comfortably under the π at which unwrapping becomes ambiguous)
for label, bad in [
    (
        "walks straight through the origin",
        np.array([1, 0.5, 0, -0.5, -1], dtype=complex),
    ),
    ("straddles the origin in a single step", np.array([1 - 1e-3j, -1 - 1e-3j])),
]:
    try:
        winding_number(bad, closed=False)
    except ValueError as e:
        print(f"caught — path {label}:\n    {e}")

# %% [markdown]
# ### Figure 7 — the loop $ABCDA$ carried to the straight world
#
# In the curved world $A\to B\to C\to D\to A$ follows grid lines once counter-clockwise around the
# centre: a **closed** loop. In the straight world the corresponding path takes three left turns,
# each leg four times the last, and does **not** close — it ends at $256\times$ where it began.
#
# That pins the start down completely. Legs $\ell,\,4i\ell,\,-16\ell,\,-64i\ell$ sum to
# $(-15-60i)\ell$, and demanding $A + (-15-60i)\ell = 256A$ gives
# $$A=\frac{(-1-4i)}{17}\,\ell .$$

# %%
ELL = 1.0
FIG7_LEGS = [
    ELL + 0j,
    4j * ELL,
    -16 * ELL + 0j,
    -64j * ELL,
]  # E, N, W, S: three left turns
A7 = ELL * (-1 - 4j) / 17
FIG7 = walk(A7, FIG7_LEGS, samples=900)

assert np.isclose(FIG7[-1], 256 * A7)
print(f"A            = {A7:+.6f}")
print(f"end of walk  = {FIG7[-1]:+.6f}")
print(f"end / A      = {(FIG7[-1] / A7).real:.9f}  = 256 ✓  (not a closed path)")
print(f"total turning / 2π = {winding_number(FIG7, closed=False):.9f}  — once around ✓")

# %% [markdown]
# Now carry it back. A branch shift of $2\pi i$ plus a blow-up by 256 is the lattice element
# $2\pi i+\log 256$, and $\beta$ maps that to $2\pi i$ exactly — so the curved-world path closes:
# $$\beta\,(\log 256+2\pi i)=\frac{\log 256+2\pi i}{\alpha}=2\pi i .$$

# %%
assert np.isclose(BETA * (LOG_S_BIG + 2j * np.pi), 2j * np.pi)
print(f"β·(log 256 + 2πi) = {BETA * (LOG_S_BIG + 2j * np.pi):.9f}   = 2πi ✓")

FIG7_W = to_curved(FIG7)
gap = abs(FIG7_W[-1] - FIG7_W[0])
print(f"curved-world loop: |w_end - w_start| = {gap:.2e}  — closed ✓")
assert gap < 1e-9

# On a *uniform* grid this path is unreadable — the first leg is 1/64 of the last. Drawn on the
# straight world's own self-similar grid, whose cells double every ring, each leg spans a comparable
# number of cells. That is exactly the property Escher was exploiting.
SQ7 = square_grid_lines(kmin=-3, kmax=7, cells=2)


def fig7_panel(extent):
    p = mv.Plane(extent=extent, grid=False)
    for pts in clip_lines(SQ7, extent):
        p.curve(pts, color=mv.FAINT, width=0.8)
    p.curve(as_xy(FIG7), color=mv.GREEN, width=2.5)
    p.points(as_xy(np.array([A7, FIG7[-1]])), color=mv.YELLOW, size=11)
    p.text([A7.real, A7.imag], "  A", color=mv.YELLOW)
    p.text([FIG7[-1].real, FIG7[-1].imag], "  256·A", color=mv.YELLOW)
    return p


print("all four legs — 1, 4, 16, 64, each turn to the left")
fig7_panel(fit_extent(FIG7)).display(format="png")
print(
    "zoomed on the start: A, the origin the path turns around, and the first two legs"
)
fig7_panel(4.5).display(format="png")

E7W = fit_extent(FIG7_W)
p = mv.Plane(extent=E7W, grid=False, axes=False)
for pts in grid_at(E7W, TILTED):
    p.curve(pts, color=mv.FAINT, width=0.8)
(
    p.curve(as_xy(FIG7_W), color=mv.GREEN, width=2.5, closed=True)
    .points(as_xy(FIG7_W[:1]), color=mv.YELLOW, size=11)
    .text([FIG7_W[0].real, FIG7_W[0].imag], "  A = A'", color=mv.YELLOW)
    .display(format="png")
)

# %% [markdown]
# ### Figures 8 and 9 — the measurement
#
# Now the converse, and the part that actually *produces* a number. Walk a square in the straight
# world starting from $A$, going up, then left, then down, then right. Both walks are closed there.
# The winding number about the origin is what separates them:
#
# * a **5×5** square misses the origin, $n=0$, so it pulls back to a closed loop (Figure 8);
# * a **7×7** square encloses it, $n=1$, so it pulls back to a path from $A$ to a *different* point
#   $A'$ — and $A/A' = \gamma$ (Figure 9).
#
# With $A = 6-6i$ the 5×5 walk spans $x\in[1,6],\,y\in[-6,-1]$ and the 7×7 walk spans
# $x\in[-1,6],\,y\in[-6,1]$: the origin sits one unit outside the first and one unit inside the
# second. That is forced — 5 and 7 differ by 2 — and it is why the paper picked those two numbers.
#
# The general statement, which the winding number gives for free: $A/A' = \gamma^{\,n}$.

# %%
A89 = 6 - 6j
walks = {}
for side in (5, 7):
    sq = walk(A89, [1j * side, -side + 0j, -1j * side, side + 0j], samples=700)
    n = winding_number(sq)
    w = to_curved(sq)
    walks[side] = (sq, w, n)
    print(
        f"\n{side}×{side} square, closed in the straight world:  winding number n = {n:+d}"
    )
    if n == 0:
        assert np.isclose(w[-1], w[0])
        print("   → closed in the curved world too:  A' = A")
    else:
        ratio = w[0] / w[-1]
        assert np.isclose(ratio, GAMMA**n)
        print(f"   → open in the curved world:  A/A' = {ratio:+.9f}")
        print(f"                        γ^{n}     = {GAMMA**n:+.9f}  ✓")
        print(
            f"      recovered |γ| = {abs(ratio):.10f},  arg γ = {np.degrees(np.angle(ratio)):.10f}°"
        )

# γ read off the 7×7 walk, to the paper's ten digits — with no reference to α at all
gamma_measured = walks[7][1][0] / walks[7][1][-1]
assert np.isclose(abs(gamma_measured), 22.5836845286, atol=1e-9)
assert np.isclose(np.degrees(np.angle(gamma_measured)), 157.6255960832, atol=1e-9)
print("\n✓ γ measured by transporting a 7×7 walk == γ derived from α in section 1")

# %% [markdown]
# The two walks in the straight world — the only difference is whether the origin falls inside:

# %%
E89 = fit_extent(walks[5][0], walks[7][0])
p = mv.Plane(extent=E89, grid=False).grid(step=1.0, alpha=0.35)
for side, col in ((5, mv.GREEN), (7, mv.RED)):
    p.curve(as_xy(walks[side][0]), color=col, width=2.5)
(
    p.points([[0, 0], [A89.real, A89.imag]], color=mv.YELLOW, size=11)
    .text([0, 0], "  origin", color=mv.YELLOW)
    .text([A89.real, A89.imag], "  A", color=mv.YELLOW)
    .display(format="png")
)

# %% [markdown]
# ...and in the curved world, on Escher's grid. The green 5×5 closes; the red 7×7 spirals in and
# stops short at $A' = A/\gamma$ — the vertex of the small central square that the paper measured.
# In the actual lithograph that point falls inside the blank circular patch, which is precisely why
# Escher never had to confront it.

# %%
EW = fit_extent(walks[5][1], walks[7][1])
p = mv.Plane(extent=EW, grid=False, axes=False)
for pts in grid_at(EW, TILTED):
    p.curve(pts, color=mv.FAINT, width=0.8)
for side, col in ((5, mv.GREEN), (7, mv.RED)):
    p.curve(as_xy(walks[side][1]), color=col, width=2.5)
w7 = walks[7][1]
(
    p.points(as_xy(np.array([w7[0], w7[-1]])), color=mv.YELLOW, size=11)
    .text([w7[0].real, w7[0].imag], "  A", color=mv.YELLOW)
    .text([w7[-1].real, w7[-1].imag], "  A' = A/γ", color=mv.YELLOW)
    .display(format="png")
)

# %% [markdown]
# ## 7. Rotation and replication — the $\gamma$ symmetry
#
# The claim to test: **rotate the lithograph clockwise by 157.6256° and shrink it by 22.5837, and it
# is the same lithograph.** Render $g(w)$, then render $g(\gamma w)$ — a window 22.58× smaller and
# turned — and compare pixel for pixel.

# %%
A = render(h, extent=1.0, res=420, hole=0.004)
B = render(lambda w: h(w * GAMMA), extent=1.0, res=420, hole=0.004)
err = np.abs(A - B).max()
print(f"max pixel difference between g(w) and g(γw): {err:.2e}")
assert err < 1e-9
print(
    "✓ g(γw) = g(w) — the picture contains itself, turned 157.6256° and shrunk 22.5837×"
)

mv.Plane(extent=1, grid=False, axes=False).raster(A).display(format="png")
mv.Plane(extent=1, grid=False, axes=False).raster(B).display(format="png")

# %% [markdown]
# ### The replication group: every fourth root of $\gamma$
#
# Because the straight picture is invariant under $z\mapsto 4z$ *and* $z\mapsto iz$, the curved one is
# invariant under multiplication by **all four fourth roots of $\gamma$** (paper, Figure 10 caption) —
# a far richer group than $\langle\gamma\rangle$, and the reason a mere 39.4° turn already brings the
# pattern back.

# %%
g4 = np.exp(LOG_GAMMA / 4)
print(f"γ^(1/4) = {abs(g4):.6f} ∠ {np.degrees(np.angle(g4)):.4f}°")
REF = render(h, extent=1.0, res=280, hole=0.006)
for k in range(4):
    m = g4 * (1j**k)
    C = render(lambda w, m=m: h(w * m), extent=1.0, res=280, hole=0.006)
    assert np.abs(C - REF).max() < 1e-9
    print(f"✓ g({abs(m):.4f} ∠{np.degrees(np.angle(m)):8.3f}° · w) = g(w)")

# %% [markdown]
# ### Figure 15 — the plate, with its centre magnified by 4 and by 16
#
# The paper's final plate shows the completed lithograph beside magnifications of its centre by 4 and
# by 16. Those are worth drawing precisely **because they are not symmetries**. The symmetry group is
# generated by $\gamma^{1/4}$ and $i$, so every element has modulus $\lvert\gamma\rvert^{k/4}=2.1800^k$
# and argument $39.4064°\,k+90°\,m$ — and a pure magnification would need that argument to vanish mod
# $360°$. The cell below searches for one and finds nothing close, so each panel really does show
# new structure rather than the picture handed back.
#
# (Empirically, not a theorem: whether $\arg\gamma/2\pi$ is irrational — which is what would rule out
# an exact hit for *every* $k$ — is not something a finite search settles.)
#
# The $\times16$ panel is the interesting near-miss: $16$ is close to $\lvert\gamma\rvert=22.58$ but
# not equal, so it comes back looking like the plate *turned by about 140°*.

# %%
# how close does any symmetry get to being a pure magnification?
cands = [
    (k, m, (k * np.degrees(np.angle(g4)) + 90 * m) % 360)
    for k in range(1, 13)
    for m in range(4)
]
k, m, deg = min(cands, key=lambda t: min(t[2], 360 - t[2]))
off = min(deg, 360 - deg)
print(
    f"searched γ^(k/4)·i^m for k ≤ 12; the closest to a pure magnification is k={k}, m={m},"
)
print(f"  which magnifies by {abs(g4) ** k:.1f} and is still {off:.2f}° off the axis")
print("→ within that range, no pure magnification is a symmetry of g\n")

# what a magnification by 16 *is*, in symmetry terms
x = np.log(16) / np.log(abs(GAMMA))
print(
    f"16 = |γ|^{x:.4f}, and γ^{x:.4f} carries a rotation of {x * np.degrees(np.angle(GAMMA)):.1f}°"
)

for e, label in [(1.0, "the plate"), (0.25, "centre ×4"), (1 / 16, "centre ×16")]:
    print(f"\n{label}")
    mv.Plane(extent=1, grid=False, axes=False).raster(
        render(h, extent=e, res=560, hole=e * 0.004)
    ).display(format="png")

# %% [markdown]
# Side by side with the symmetry test of the previous cells, the contrast is the whole point:
# multiplying by $\gamma$ changes nothing, magnifying by 4 changes everything.

# %%
ref = render(h, extent=1.0, res=280, hole=0.006)
by_gamma = render(lambda w: h(w * GAMMA), extent=1.0, res=280, hole=0.006)
by_four = render(h, extent=0.25, res=280, hole=0.0015)
d_gamma = np.abs(ref - by_gamma).mean()
d_four = np.abs(ref - by_four).mean()
print(
    f"mean |g − g∘(×γ)|        = {d_gamma:.2e}   ← a symmetry: the picture is unchanged"
)
print(f"mean |g − g magnified ×4| = {d_four:.4f}     ← not a symmetry: new content")
assert d_gamma < 1e-9 < 0.01 < d_four

# %% [markdown]
# ### Replication as motion: the Droste zoom
#
# Zoom by $\gamma^{t}$ for $t\in[0,1]$ and you arrive exactly where you started — a seamless loop.
# This is the animation on `escherdroste.math.leidenuniv.nl`, and the payoff shot of the video.


# %%
def zoom_frame(t):
    """Frame builder for `mv.scrubber` / `mv.to_gif` — returns a Plane, which they rasterize."""
    m = np.exp(LOG_GAMMA * t)
    p = mv.Plane(extent=1, grid=False, axes=False)
    p.view.size = (420, 420)
    return p.raster(render(lambda w: h(w * m), extent=1.0, res=360, ss=2, hole=0.005))


try:  # a Play button + slider; frames are pre-rendered so dragging is instant
    from IPython.display import display

    display(
        mv.scrubber(zoom_frame, n=24, interval=90, label="t  (zoom by γᵗ)", width=420)
    )
except Exception as e:  # headless, or no ipywidgets
    print("scrubber unavailable here:", type(e).__name__, e)


# %% [markdown]
# For a version that renders on GitHub and survives a headless `jupytext --execute`:
#
# ```python
# mv.to_gif(zoom_frame, "droste_zoom.gif", n=36, fps=18)
# ```

# %% [markdown]
# ## 8. Variants: other $\delta$ (the paper's closing paragraph)
#
# > *"Other complex analytic maps $h:\mathbb{C}^*/\langle\delta\rangle\to\mathbb{C}^*/\langle 256\rangle$
# > for various $\delta$ give rise to interesting variants of Prentententoonstelling."*
#
# The derivation used only that one counter-clockwise loop maps to $2\pi i+\log 256$. Send it instead
# to $2\pi i\,n+\log 256$ — wind $n$ times — and you get a family. $n=1$ is Escher's;
# $n=0$ is degenerate ($\gamma=1$: the loop carries no rotation, so there is no elliptic curve left);
# large $\lvert n\rvert$ unwinds towards the undistorted picture.


# %%
def alpha_variant(n=1, s=S_BIG):
    """α for the map sending one loop to n loops plus a blow-up by s."""
    return (2j * np.pi * n + np.log(s)) / (2j * np.pi)


for n in (1, 2, 3, -1):
    a = alpha_variant(n)
    gam = np.exp(np.log(S_BIG) / a)
    print(
        f"n={n:+d}:  α = {a.real:+.4f}{a.imag:+.4f}i   |γ| = {abs(gam):8.4f}   ∠{np.degrees(np.angle(gam)):8.3f}°"
    )
    mv.Plane(extent=1, grid=False, axes=False).raster(
        render(
            lambda w, a=a: np.exp(a * np.log(w)), extent=1.0, res=380, ss=2, hole=0.005
        )
    ).display(format="png")

# %% [markdown]
# ## 9. Using your own image
#
# The only requirement is that the drawing be **seamless across the fold** — nothing may touch
# $L(z)=1$ or $L(z)=4$, or $f(4z)=f(z)$ fails and a seam spirals through the result. Inset your photo
# in a plain margin (the margin is the gallery wall) and everything above re-runs unchanged:
#
# ```python
# from PIL import Image
#
# def gallery_from_file(path, px=1000, centre=2.5, half=1.0):
#     """Hang one image on the wall at (centre, 0) and let the ×i symmetry make four."""
#     img = np.asarray(Image.open(path).convert("RGB")) / 255.0
#     out = np.tile(np.array(mv.palette.rgb01(WALL)), (px, px, 1))
#     lo = int((centre - half + MOTIF_EXTENT) / (2 * MOTIF_EXTENT) * px)
#     hi = int((centre + half + MOTIF_EXTENT) / (2 * MOTIF_EXTENT) * px)
#     mid = px // 2
#     n = hi - lo
#     patch = np.asarray(Image.open(path).convert("RGB").resize((n, n))) / 255.0
#     out[mid - n // 2 : mid - n // 2 + n, lo:hi] = patch[::-1]
#     for k in (1, 2, 3):                      # enforce f(iz) = f(z) exactly
#         out = np.maximum(out, np.rot90(out, k, axes=(0, 1))) if False else out
#     acc = sum(np.rot90(out, k, axes=(0, 1)) for k in range(4)) - 3 * np.tile(
#         np.array(mv.palette.rgb01(WALL)), (px, px, 1))
#     return np.clip(acc, 0, 1)
#
# GALLERY = gallery_from_file("my_photo.jpg")   # then re-run sections 2–8
# assert abs(centre) - half > 1 and abs(centre) + half < 4   # stay inside the ring!
# ```
#
# Two knobs worth turning:
#
# * `S = 256.0` instead of `4.0` — a *single* study rather than four. The picture then only repeats
#   after a full turn, and only $\gamma$ (not its fourth roots) is a symmetry.
# * The margin. Shrink it until the frame touches $L=4$ and watch the seam appear: it is the clearest
#   possible demonstration of what the periodicity is actually doing.

# %% [markdown]
# ## 10. The rest of the paper — what each remaining figure would take
#
# Everything above is built from `f_straight`, `h`, and `render`. Here is what the paper's other
# figures need on top of that, roughly in order of effort.
#
# ### Done in section 5a — Escher's false starts (Figures 2 and 3)
#
# Built at the top of section 5: the same vertex lattice joined by straight chords and then
# conformally, with the corner error that separates them measured.
#
# ### Done in section 6 — loop transport (Figures 7, 8, 9)
#
# Section 6 builds these: the winding-number integral, the three-left-turns path of Figure 7, and the
# 5×5 / 7×7 walks of Figures 8 and 9 that let you read γ straight off the picture.
#
# ### Done in section 7 — Figure 15's magnification triptych
#
# Built at the end of section 7: the plate beside its centre magnified by 4 and 16, with the check
# that no pure magnification is a symmetry (so each panel is new content, not a repeat).
#
# ### Needs a fit, not a formula — Figure 4, Escher's *actual* grid
#
# Section 5 draws the *idealized* grid. Escher's own grid is close but not conformal: the paper notes
# his central square is larger than theirs, i.e. his effective $|\gamma|$ is nearer 20 than 22.58. To
# reproduce Figure 4 you would have to digitize the grid off the plate and fit a (non-conformal)
# deformation to it — there is no closed form, because the whole point is that Escher missed. A
# cheaper and more honest version: parameterize
# `alpha_from_gamma = lambda g: LOG_S_BIG / np.log(g)` and render $g$ at Escher's measured
# $\gamma \approx 20e^{3i}$ next to the exact one, to see how much the discrepancy actually shows.
#
# ### Blocked on the artwork itself (Figures 1, 5, 12, 13)
#
# The lithograph, Escher's studies, the rectified lithograph, and the Richter–Hofstra completion are
# all reproductions of copyrighted work (Cordon Art B.V.). The *procedure* behind Figure 12 —
# "rectify the lithograph by means of his own grid" — is just `render` with the map inverted,
# `zmap = lambda z: np.exp(BETA * np.log(z))`, sampling the lithograph instead of `GALLERY`; the
# obstacle is the source image, not the mathematics. Section 9 shows how to substitute your own.
#
# ### Beyond the paper
#
# * A **hyperbolic** variant: replace $\mathbb{C}^*/\langle\delta\rangle$ with a quotient by a Fuchsian
#   group and you get Escher's *Circle Limit* family instead — the same "quotient by a discrete group,
#   lift, transport, push forward" pattern, different group.
# * The **elliptic curve** itself: $\mathbb{C}/L_\gamma$ is a torus, and section 3's log tile is its
#   fundamental domain. Gluing that tile into an actual torus surface is a job for
#   `mv.Space3D(...).surface(...)`, and would make the phrase "drawn on an elliptic curve" literal.
#
# %% [markdown]
# ## Recap
#
# | object | formula | verified above |
# |---|---|---|
# | straight picture | $f$ on the square annulus $1\le L(z)<4$, folded | $f(4z)=f(iz)=f(256z)=f(z)$ |
# | log picture | $f(e^{\xi})$ | doubly periodic, $\log 256 \times 2\pi i$ |
# | the exponent | $\alpha=(2\pi i+\log 256)/(2\pi i)$ | $=1-0.8825424006\,i$ |
# | Escher's map | $h(w)=w^{\alpha}$ | $\gamma^{\alpha}=256$, $i^{\alpha}=4i$ |
# | complex period | $\log\gamma=(\log 256)/\alpha$ | $\lvert\gamma\rvert=22.5836845286$, $\arg\gamma=157.6255960832°$ |
# | the lithograph | $g=f\circ h$ | $g(\gamma w)=g(w)$, and for every fourth root of $\gamma$ |
# | the grid | lattice $\times\,\beta=1/\alpha$, then $\exp$ | rotate $41.43°$, scale $0.7498$ |

# %% [markdown]
#

# %% [markdown]
#
