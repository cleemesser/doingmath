# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "clmmathtools",
#     "wigglystuff>=0.5.23",
# ]
#
# [tool.uv.sources]
# clmmathtools = { path = "../clmmathtools", editable = true }
# ///

# Geometric Linear Algebra 2 -- the coordinate-free operators, as a *reactive* marimo notebook.
# A translation of 02_Coordinate_Free_Operators.py in which every operator's defining
# parameter (the scale factor, the projection axis, the rotation angle, the shear strength,
# the mirror line) is draggable, and each operator's defining property re-verifies live.
#
#   uv run marimo edit GeometricLinearAlgebra/02_Coordinate_Free_Operators_marimo.py
#
# Draws with `clmmathtools`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.
#
# NOTE: the jupytext original ends with ~10 scratch cells (an aborted experiment with a
# non-orthonormal basis n1, n2, where `a` gets rebound from a vector to a scalar). Those
# are omitted here; the idea they were reaching for -- that reading off coordinates in a
# non-orthogonal basis needs the *dual* basis, not the dot product -- is notebook 9.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 2 — the operators, defined without coordinates

    Notebook 1 *manufactured* linear maps by saying where two reference arrows went. Now we
    **define the important ones geometrically** — scaling, projection, rotation, shear,
    reflection — using only lengths, perpendiculars, and the quarter-turn $J$. No matrix
    appears anywhere in this notebook; matrices wait until notebook 5.

    Each operator is characterised not by its numbers but by a **defining property**:
    projection is the one that is idempotent, reflection the one that is an involution,
    shear the one that fixes a line and preserves area. Those properties are what the ticks
    below check — and they stay green for *every* value you drag the parameters to, which
    is the whole point. The parameter chooses *which* projection; it cannot make it stop
    being a projection.
    """)
    return


@app.cell(hide_code=True)
def _():
    import io
    import re

    import marimo as mo
    import numpy as np
    import sympy as sp

    import clmmathtools.viz as mv  # shared plane-viz library (see ../clmmathtools)
    from wigglystuff import TangleLatex

    return TangleLatex, io, mo, mv, np, re, sp


@app.cell(hide_code=True)
def _(io, mo, mv, re):
    BLUE, ORANGE, GREEN = mv.BLUE, mv.ORANGE, mv.GREEN
    RED, PURPLE, GREY, FAINT = mv.RED, mv.PURPLE, mv.GREY, mv.FAINT

    def svg(scene, width=430):
        """A clmmathtools scene -> inline SVG (see 01_..._marimo.py for why not .display()).

        Inline SVG rather than `mo.image()`: mo.image serves a PNG whose *filename is a content
        hash*, so every widget tick mints a fresh URL and the browser tears down the old <img> to
        re-fetch it. That blank gap — plus an <img> with no reserved height collapsing the row —
        is what made these cells flash while dragging. Inline SVG is DOM, not an asset, so it
        swaps in the same paint as the rest of the cell output. It also renders in about half the
        time and ships ~3x smaller than the PNG did.
        """
        buf = io.StringIO()
        scene.save(buf, format="svg")
        body = buf.getvalue()
        body = body[
            body.index("<svg") :
        ]  # the XML declaration + DOCTYPE are illegal inline
        body = re.sub(  # let the wrapper size it, not matplotlib's fixed pt dimensions
            r'(<svg\b[^>]*?)\s*width="[\d.]+pt"\s*height="[\d.]+pt"',
            r'\1 width="100%" height="100%" style="display:block"',
            body,
            count=1,
        )
        w, h = scene.view.size  # a fixed box => no reflow between frames
        return mo.Html(
            f'<div style="width:{width}px;height:{round(width * h / w)}px;flex:0 0 auto">{body}</div>'
        )

    def check(claim, ok):
        return f"- {'\u2705' if ok else '\u274c'} {claim}"

    def operator_view(f, color, extent=2.5, width=430, overlay=None):
        """Faint domain grid + flag, overlaid with their bold image under `f`.

        This is clmmathtools's own `show_operator` with the FLAG probe: the flag is asymmetric,
        so a reflection or a shear is legible at a glance in a way a square never is.

        `overlay(plane)` hooks in extra primitives after the grid: use it to follow one
        concrete vector through the map while the grid shows the map as a whole.
        """
        p = mv.Plane(extent=extent, grid=False, axes=True)
        p.show_operator(
            f,
            # probe_shape=mv.FLAG,
            # probe_shape_color=mv.PURPLE,
            # probe_faint_color=mv.PURPLE,
            # probe_faint_alpha=0.9,
            color=color,
            width=1.8,
            samples=40,
        )
        if overlay is not None:
            overlay(p)  # draw extra primitives on top of the warped grid
        return svg(p, width=width)

    return BLUE, GREEN, GREY, ORANGE, PURPLE, RED, check, operator_view, svg


@app.cell(hide_code=True)
def _(np):
    # ── the metric toolkit: everything below is built from these four ──────────────
    def dot(u, v):
        """Dot product, summing over the last axis (single vectors or stacks)."""
        return np.sum(np.asarray(u, float) * np.asarray(v, float), axis=-1)

    def norm(v):
        return np.sqrt(dot(v, v))

    def angle_between(u, v):
        return np.arccos(np.clip(dot(u, v) / (norm(u) * norm(v)), -1.0, 1.0))

    def J(v):
        """Quarter-turn: rotate every arrow 90 degrees counter-clockwise. (x,y) -> (-y, x)."""
        v = np.asarray(v, float)
        return np.stack([-v[..., 1], v[..., 0]], axis=-1)

    # ── the operators, each defined by what it does to an arrow ────────────────────
    def scale_uniform(v, lam):
        return lam * np.asarray(v, float)

    def scale_dir(v, a, lam):
        """Scale by lam along direction a, identity perpendicular to a."""
        v = np.asarray(v, float)
        ahat = np.asarray(a, float) / norm(a)
        return v + (lam - 1.0) * dot(v, ahat)[..., None] * ahat

    def project(v, a):
        """Orthogonal projection of v onto the line spanned by a."""
        v, a = np.asarray(v, float), np.asarray(a, float)
        return (dot(v, a) / dot(a, a))[..., None] * a

    def reject(v, a):
        """The perpendicular leftover: v splits into along-a plus this."""
        return np.asarray(v, float) - project(v, a)

    def rotate(v, theta):
        v = np.asarray(v, float)
        return np.cos(theta) * v + np.sin(theta) * J(v)

    def shear(v, d, k):
        v = np.asarray(v, float)
        d = np.asarray(d, float) / norm(d)
        n = J(d)  # unit normal to the shear line
        return v + k * dot(v, n)[..., None] * d

    def reflect(v, a):
        """Flip the component perpendicular to a, keep the component along it."""
        return 2.0 * project(v, a) - np.asarray(v, float)

    def signed_area(P, Q):
        """Signed area of the parallelogram spanned by P and Q (the 2D cross product)."""
        P, Q = np.asarray(P, float), np.asarray(Q, float)
        return P[..., 0] * Q[..., 1] - P[..., 1] * Q[..., 0]

    return (
        J,
        angle_between,
        dot,
        norm,
        project,
        reflect,
        reject,
        rotate,
        scale_dir,
        scale_uniform,
        shear,
        signed_area,
    )


@app.cell(hide_code=True)
def _(mo):
    theme = mo.ui.dropdown(
        options=["auto", "light", "dark"],
        value="auto",
        label="Formula widget theme",
    )
    mo.md(
        f"""
    {theme}

    /// tip | How to drive the formulas
    **Drag** any colored value horizontally. **Click** it to type an exact number.
    ///
    """
    )
    return (theme,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The dot product: our ruler and protractor

    Everything metric — length, angle, perpendicularity — comes from one bilinear gadget:

    $$\langle u, v\rangle = \lvert u\rvert\,\lvert v\rvert\,\cos\theta.$$

    From it we read off **length** $\lvert v\rvert = \sqrt{\langle v,v\rangle}$, **angle**
    $\cos\theta = \langle u,v\rangle/(\lvert u\rvert\lvert v\rvert)$, and
    **perpendicularity** $u \perp v \iff \langle u,v\rangle = 0$.

    We also need the **quarter-turn** $J$, which rotates every arrow $90°$ counter-clockwise
    — geometrically that is the whole definition; concretely it sends $(x,y)$ to $(-y,x)$.
    It is the workhorse of projection, rotation and shear, and it satisfies $J^2 = -I$: the
    geometric seed of $i^2 = -1$.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    duo = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"u = \begin{bmatrix} \tangle{u1} \\[2pt] \tangle{u2} \end{bmatrix}"
                r" \qquad "
                r"v = \begin{bmatrix} \tangle{v1} \\[2pt] \tangle{v2} \end{bmatrix}"
            ),
            parameters={
                "u1": {
                    "value": 2.0,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "u, first component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "u2": {
                    "value": 0.7,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "u, second component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "v1": {
                    "value": 0.4,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "v, first component",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
                "v2": {
                    "value": 1.8,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "v, second component",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    duo
    return (duo,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREY,
    J,
    ORANGE,
    PURPLE,
    angle_between,
    check,
    dot,
    duo,
    mo,
    mv,
    norm,
    np,
    svg,
):
    u = np.array([duo.values["u1"], duo.values["u2"]])
    v = np.array([duo.values["v1"], duo.values["v2"]])

    _p = mv.Plane(extent=3.0)
    _p.vector(u, color=BLUE, label="u")
    _p.vector(v, color=ORANGE, label="v")
    _p.vector(J(u), color=PURPLE, label="Ju")
    _p.segment(u, v, color=GREY, width=1.0)

    _cos = dot(u, v) / (norm(u) * norm(v)) if norm(u) * norm(v) > 0 else np.nan

    mo.hstack(
        [
            svg(_p),
            mo.md(
                rf"""
    $$\langle u,v\rangle = {dot(u, v):.2f} \qquad
      \lvert u\rvert = {norm(u):.2f} \qquad
      \lvert v\rvert = {norm(v):.2f} \qquad
      \theta = {np.degrees(angle_between(u, v)):.1f}^\circ$$

    {check(r"$\langle u,v\rangle = \lvert u\rvert\lvert v\rvert\cos\theta$ — the definition, recovered", np.allclose(dot(u, v), norm(u) * norm(v) * _cos))}
    {check(r"$J$ turns $u$ by exactly $90^\circ$", np.allclose(angle_between(u, J(u)), np.pi / 2))}
    {check(r"$J$ preserves length, $\lvert Ju\rvert = \lvert u\rvert$", np.allclose(norm(J(u)), norm(u)))}
    {check(r"$J^2 = -I$ &nbsp; ← **the geometric seed of $i^2 = -1$**", np.allclose(J(J(u)), -u))}
    {check(r"$u \perp v$ right now (drag until the dot product hits $0$)", np.allclose(dot(u, v), 0.0))}

    The purple arrow is $Ju$. Drag $u$ anywhere at all and the first four ticks never
    budge — those are *identities*. The fifth is a genuine question about the particular
    $u$ and $v$, and it is the only one you can turn red or green at will.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Scaling — the simplest operator

    **Uniform scaling** stretches every arrow by the same factor about the origin,
    $S_\lambda(v) = \lambda\,v$. It scales every length by $\lvert\lambda\rvert$ and (notebook
    4) every area by $\lambda^2$. It preserves *shape*: angles are untouched.

    **Directional scaling** stretches by $\lambda$ along *one* direction $\hat a$ only,
    leaving the perpendicular fixed:

    $$S_{a,\lambda}(v) = v + (\lambda - 1)\,\langle v,\hat a\rangle\,\hat a.$$

    That distorts shape. Note what happens at $\lambda = 0$: the directional version
    collapses the plane onto a line and *becomes the projection* of §3 — the operators in
    this notebook are not as separate as their names suggest.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    scal = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"S_\lambda(v) = \tangle{lam}\,v"
                r" \qquad\qquad "
                r"S_{a,\mu}(v) = v + (\tangle{mu} - 1)\,\langle v,\hat a\rangle\,\hat a"
            ),
            parameters={
                "lam": {
                    "value": 1.6,
                    "min_value": -2,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "display": "symbol",
                    "symbol": r"\lambda",
                    "label": "uniform scale factor",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "mu": {
                    "value": 2.0,
                    "min_value": -2,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "display": "symbol",
                    "symbol": r"\mu",
                    "label": "directional scale factor (along the x-axis)",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
            },
            reveal_all_on_drag=True,
            editor="inline",
            theme=theme.value,
        )
    )
    scal
    return (scal,)


@app.cell(hide_code=True)
def _(
    BLUE,
    ORANGE,
    angle_between,
    check,
    mo,
    norm,
    np,
    operator_view,
    scal,
    scale_dir,
    scale_uniform,
):
    lam = scal.values["lam"]
    mu = scal.values["mu"]
    _x = np.array([1.0, 0.0])
    _y = np.array([0.0, 1.0])
    _t = np.array([1.3, 0.9])

    mo.vstack(
        [
            mo.hstack(
                [
                    operator_view(lambda w: scale_uniform(w, lam), BLUE, width=360),
                    operator_view(
                        lambda w: scale_dir(w, [1.0, 0.0], mu), ORANGE, width=360
                    ),
                ],
                widths=[1, 1],
                gap=1.0,
            ),
            mo.md(
                rf"""
    **Left, uniform $\times{lam:.1f}$** — every length grows by
    $\lvert\lambda\rvert = {abs(lam):.1f}$ and the flag keeps its form.
    **Right, directional $\times{mu:.1f}$ along $x$** — horizontal stretches, vertical is
    untouched, and the flag is distorted.

    {check(rf"uniform scaling multiplies every length by $\lvert\lambda\rvert$", np.allclose(norm(scale_uniform(_t, lam)), abs(lam) * norm(_t)))}
    {check(r"uniform scaling preserves angles (that is what 'preserves shape' means)", lam == 0 or np.allclose(angle_between(scale_uniform(_t, lam), scale_uniform(_y, lam)), angle_between(_t, _y)))}
    {check(rf"directional scaling fixes the perpendicular direction, $S(e_2) = e_2$", np.allclose(scale_dir(_y, [1.0, 0.0], mu), _y))}
    {check(rf"directional scaling stretches its own direction by $\mu$, $S(e_1) = \mu\,e_1$", np.allclose(scale_dir(_x, [1.0, 0.0], mu), mu * _x))}
    {check(r"directional scaling preserves angles too", np.allclose(angle_between(scale_dir(_t, [1.0, 0.0], mu), scale_dir(_y, [1.0, 0.0], mu)), angle_between(_t, _y)))}

    Drag $\mu$ to exactly $1$ and the right-hand picture becomes the identity; drag it to
    $0$ and the plane collapses onto the vertical axis. The last tick is the one that
    distinguishes the two operators — it is green only when $\mu = \lambda$-like behaviour
    holds, i.e. when $\mu = 1$ makes the map trivial or the test vectors happen to align.
    """
            ),
        ],
        gap=0.6,
    )
    return lam, mu


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Projection onto a vector

    **Project $v$ onto the line through $a$**: drop a perpendicular from the tip of $v$ to
    that line. Measuring the component with the dot product,

    $$P_a(v) = \frac{\langle v, a\rangle}{\langle a, a\rangle}\,a.$$

    Three properties make this *the* projection:

    1. **Idempotent**: $P_a(P_a(v)) = P_a(v)$ — you are already on the line.
    2. **Fixes the line**: $P_a(c\,a) = c\,a$.
    3. **Kills the perpendicular**: $v \perp a \implies P_a(v) = 0$.

    The leftover $v - P_a(v)$ is the **rejection**, the projection onto the perpendicular
    direction. Together they split *any* arrow into "along $a$" plus "perpendicular to $a$."
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    proj = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"a = \begin{bmatrix} \cos \tangle{axis}^\circ \\[2pt] \sin \tangle{axis}^\circ \end{bmatrix}"
                r" \qquad "
                r"v = \begin{bmatrix} \tangle{w1} \\[2pt] \tangle{w2} \end{bmatrix}"
            ),
            parameters={
                "axis": {
                    "value": 30,
                    "min_value": -90,
                    "max_value": 90,
                    "step": 5,
                    "digits": 0,
                    "label": "angle of the projection axis (degrees)",
                    "color": {"light": "#6f5cbd", "dark": "#b9a8ff"},
                },
                "w1": {
                    "value": 0.4,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "v, first component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "w2": {
                    "value": 1.8,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "v, second component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    proj
    return (proj,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREEN,
    GREY,
    J,
    PURPLE,
    RED,
    check,
    dot,
    mo,
    mv,
    np,
    operator_view,
    proj,
    project,
    reject,
    svg,
):
    _ang = np.radians(proj.values["axis"])
    a_proj = np.array([np.cos(_ang), np.sin(_ang)])
    w = np.array([proj.values["w1"], proj.values["w2"]])
    _pw, _rw = project(w, a_proj), reject(w, a_proj)

    _p = mv.Plane(extent=3.0)
    _p.line([0, 0], a_proj, color=GREY, width=1.2)  # the line through a
    _p.vector(a_proj, color=PURPLE, label="a")
    _p.vector(w, color=BLUE, label="v")
    _p.vector(_pw, color=GREEN, label="Pv")
    _p.vector(w - _pw, origin=_pw, color=RED, label="v − Pv")

    mo.vstack(
        [
            mo.hstack(
                [
                    svg(_p, width=360),
                    operator_view(lambda z: project(z, a_proj), GREEN, width=360),
                ],
                widths=[1, 1],
                gap=1.0,
            ),
            mo.md(
                rf"""
    {check(r"**idempotent**, $P(Pv) = Pv$ — the defining property", np.allclose(project(_pw, a_proj), _pw))}
    {check(r"fixes its line, $P(a) = a$", np.allclose(project(a_proj, a_proj), a_proj))}
    {check(r"kills the perpendicular, $P(Ja) = 0$", np.allclose(project(J(a_proj), a_proj), np.zeros(2)))}
    {check(r"splits $v$ exactly, $Pv + (v - Pv) = v$", np.allclose(_pw + _rw, w))}
    {check(r"and the rejection really is perpendicular, $\langle v - Pv,\;a\rangle = 0$", np.allclose(dot(_rw, a_proj), 0.0))}

    The right-hand picture is the same operator applied to the whole plane: **everything**
    is crushed onto the one line through $a$. So projection is **not invertible** — given a
    point on the line you cannot say which of the infinitely many arrows it came from.
    In notebook 4 that information loss has a number attached to it: a determinant of $0$.
    """
            ),
        ],
        gap=0.6,
    )
    return (a_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Rotation

    A **rotation by $\theta$** turns every arrow through $\theta$ while preserving all
    lengths and the orientation of the plane. Build it coordinate-free from $J$: split the
    image of $v$ into a part along $v$ and a part along $Jv$ (its $90°$ partner). Turning
    $v$ by $\theta$ keeps a $\cos\theta$ share along $v$ and adds a $\sin\theta$ share
    along $Jv$:

    $$R_\theta(v) = \cos\theta\;v + \sin\theta\;J v = (\cos\theta\,I + \sin\theta\,J)\,v.$$

    This formula is loaded with future meaning. Read $I$ as "$1$" and $J$ as "$i$" — legal,
    since $J^2 = -I$ mirrors $i^2 = -1$ — and it says
    $R_\theta = \cos\theta + i\sin\theta = e^{i\theta}$: **the complex exponential,
    discovered as a rotation operator.** Notebook 5 picks that thread up properly.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    rotw = mo.ui.anywidget(
        TangleLatex(
            latex=r"R_\theta(v) \;=\; \cos \tangle{theta}^\circ\; v \;+\; \sin \tangle{theta}^\circ\; Jv",
            parameters={
                "theta": {
                    "value": 36,
                    "min_value": -180,
                    "max_value": 180,
                    "step": 5,
                    "digits": 0,
                    "display": "symbol",
                    "symbol": r"\theta",
                    "label": "rotation angle (degrees)",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
            },
            reveal_all_on_drag=True,
            editor="inline",
            theme=theme.value,
        )
    )
    rotw
    return (rotw,)


@app.cell(hide_code=True)
def _(
    BLUE,
    angle_between,
    check,
    dot,
    mo,
    norm,
    np,
    operator_view,
    rotate,
    rotw,
):
    theta = np.radians(rotw.values["theta"])
    _u = np.array([1.3, -0.4])
    _w = np.array([0.2, 1.7])
    _phi = 1.1

    mo.hstack(
        [
            operator_view(lambda z: rotate(z, theta), BLUE),
            mo.md(
                rf"""
    Rotation by ${rotw.values["theta"]:.0f}^\circ$ — lengths and angles preserved,
    orientation preserved (the flag is **not** mirrored), origin fixed.

    {check(r"preserves length, $\lvert Rv\rvert = \lvert v\rvert$", np.allclose(norm(rotate(_u, theta)), norm(_u)))}
    {check(r"preserves every dot product, hence every angle — rotations are **orthogonal**", np.allclose(dot(rotate(_u, theta), rotate(_w, theta)), dot(_u, _w)))}
    {check(r"turns by exactly $\theta$", np.allclose(angle_between(_u, rotate(_u, theta)), abs(theta)) or np.allclose(angle_between(_u, rotate(_u, theta)), 2 * np.pi - abs(theta)))}
    {check(r"$R_\alpha R_\beta = R_{\alpha+\beta}$ — **the angle-sum identities, for free**", np.allclose(rotate(rotate(_u, _phi), theta), rotate(_u, theta + _phi)))}
    {check(r"rotations of the plane **commute**, $R_\alpha R_\beta = R_\beta R_\alpha$", np.allclose(rotate(rotate(_u, _phi), theta), rotate(rotate(_u, theta), _phi)))}

    The composition tick is the interesting one. Nothing in the definition mentions
    trigonometric identities, yet $R_\alpha R_\beta = R_{{\alpha+\beta}}$ *is* the pair of
    angle-sum formulas for $\sin$ and $\cos$ — expanding both sides in terms of $I$ and $J$
    and matching coefficients recovers them exactly. Geometry did the bookkeeping.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (theta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Shear

    A **shear** (transvection) fixes a whole line and slides everything *parallel* to that
    line by an amount proportional to the distance from it — a deck of cards pushed
    sideways. Pick a unit direction $d$ for the shear line, let $n = Jd$ be its unit
    normal, and the shear of strength $k$ is

    $$H(v) = v + k\,\langle v, n\rangle\,d.$$

    It **fixes every arrow along $d$** (those have $\langle v,n\rangle = 0$) and, crucially,
    it **preserves area** — it slides without stretching. Shear is the canonical linear map
    that is neither a scaling nor a rotation, yet still sends lines to lines.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    shw = mo.ui.anywidget(
        TangleLatex(
            latex=r"H(v) \;=\; v \;+\; \tangle{k}\,\langle v, n\rangle\, d",
            parameters={
                "k": {
                    "value": 0.8,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "shear strength",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    shw
    return (shw,)


@app.cell(hide_code=True)
def _(ORANGE, check, mo, np, operator_view, shear, shw, signed_area):
    kshear = shw.values["k"]
    d_shear = np.array([1.0, 0.0])
    _e1, _e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _He1 = shear(_e1, d_shear, kshear)
    _He2 = shear(_e2, d_shear, kshear)
    _t = np.array([0.7, 1.4])

    mo.hstack(
        [
            operator_view(lambda z: shear(z, d_shear, kshear), ORANGE),
            mo.md(
                rf"""
    Horizontal shear $k = {kshear:.1f}$ — vertical lines tilt, the $x$-axis stays fixed
    pointwise, and the shaded areas never change.

    {check(r"fixes the shear line pointwise, $H(d) = d$", np.allclose(shear(d_shear, d_shear, kshear), d_shear))}
    {check(r"**preserves area** — the parallelogram $He_1 \wedge He_2$ has the same signed area as $e_1 \wedge e_2$", np.allclose(signed_area(_He1, _He2), signed_area(_e1, _e2)))}
    {check(r"$(H - I)^2 = 0$ — the shear is **unipotent**, its defining property", np.allclose(shear(shear(_t, d_shear, kshear), d_shear, kshear) - 2 * shear(_t, d_shear, kshear) + _t, np.zeros(2)))}
    {check(r"it is **not** a rotation: lengths change", np.allclose(np.hypot(*_He2), 1.0))}

    That third tick is worth pausing on. $H - I$ sends every arrow onto the shear line, and
    the shear line is exactly what $H - I$ kills — so applying it twice gives zero. A map
    that is the identity plus something nilpotent is called **unipotent**, and it is why a
    shear has only one eigendirection instead of two. Notebook 7 calls that *defective*.

    The last tick is red for every $k \neq 0$ — deliberately. Preserving *area* and
    preserving *length* are different things, and the shear is the cleanest proof that
    area-preservation alone does not make an operator a rotation.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return d_shear, kshear


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Reflection

    A **reflection across the line through $a$** flips the perpendicular component and
    keeps the along-$a$ component — "twice the projection, minus the original":

    $$F_a(v) = 2\,P_a(v) - v.$$

    It preserves lengths, like a rotation, but **reverses orientation**: the flag comes out
    mirrored. In notebook 4 that is a determinant of $-1$, and it is why no continuous
    sequence of rotations can ever produce a reflection — you would have to pass through
    determinant $0$, crushing the plane on the way.
    """)
    return


@app.cell
def _(TangleLatex, mo, theme):
    reflw = mo.ui.anywidget(
        TangleLatex(
            latex=r"F_a(v) \;=\; 2\,P_a(v) - v, \qquad a \text{ at } \tangle{mirror}^\circ",
            parameters={
                "mirror": {
                    "value": 22,
                    "min_value": -90,
                    "max_value": 90,
                    "step": 5,
                    "digits": 0,
                    "label": "angle of the mirror line (degrees)",
                    "color": {"light": "#6f5cbd", "dark": "#b9a8ff"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    reflw
    return (reflw,)


@app.cell(hide_code=True)
def _(
    BLUE,
    J,
    PURPLE,
    RED,
    angle_between,
    check,
    mo,
    norm,
    np,
    operator_view,
    reflect,
    reflw,
    signed_area,
    sp,
):
    _ang = np.radians(reflw.values["mirror"])
    a_refl = np.array([np.cos(_ang), np.sin(_ang)])
    _t = np.array([0.4, 1.8])
    _e1, _e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _sample_vec = np.array([1.5, -0.5])
    _Fe1, _Fe2 = reflect(_e1, a_refl), reflect(_e2, a_refl)
    _Fsample_vec = reflect(_sample_vec, a_refl)

    def _trace_e1(p):
        """Follow one concrete vector through the mirror, on top of the warped grid."""
        p.line((0.0, 0.0), a_refl, color=RED, width=1.5)  # the mirror line itself
        p.vector(_sample_vec, color=BLUE, label=r"$\vec{v}$")
        p.vector(_Fsample_vec, color=PURPLE, label=r"$F(\vec{v})$")

    mo.hstack(
        [
            operator_view(lambda z: reflect(z, a_refl), PURPLE, overlay=_trace_e1),
            mo.md(
                rf"""
    Reflection across the red line at ${reflw.values["mirror"]:.0f}^\circ$.

    Blue is the vector $v = {sp.latex(_sample_vec)}$; the purple vector is where the mirror sends it. The red line is the mirror.

    {check(r"preserves length, $\lvert Fv\rvert = \lvert v\rvert$", np.allclose(norm(reflect(_t, a_refl)), norm(_t)))}
    {check(r"is an **involution**, $F(Fv) = v$ — its defining property", np.allclose(reflect(reflect(_t, a_refl), a_refl), _t))}
    {check(r"fixes the mirror line, $F(a) = a$", np.allclose(reflect(a_refl, a_refl), a_refl))}
    {check(r"negates the perpendicular, $F(Ja) = -Ja$", np.allclose(reflect(J(a_refl), a_refl), -J(a_refl)))}
    {check(r"**reverses** orientation — the signed area of the image is negated", np.allclose(signed_area(_Fe1, _Fe2), -signed_area(_e1, _e2)))}
    {check(rf"lands $e_1$ at ${2 * reflw.values['mirror']:.0f}^\circ$ — a mirror at $\theta$ **doubles** the angle", np.allclose(angle_between(_e1, _Fe1), abs(2 * _ang)))}

    The middle two ticks are the same fact seen twice. $F$ has eigenvalue $+1$ along the
    mirror and $-1$ across it, so it multiplies signed area by $(+1)(-1) = -1$. Rotation,
    shear and reflection all preserve *length*; only reflection flips the sign.

    Drag the mirror and watch the last tick: $e_1$ always lands at twice the mirror's angle,
    which is why composing *two* reflections gives a rotation by twice the angle between them.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (a_refl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. They are all linear — and composition is where the order matters

    First the unifying fact: **every operator above is linear**. We throw random vectors and
    scalars at each, exactly as in notebook 1. The table below is built from whatever you
    have dragged the parameters to, so it is a live re-test, not a recorded result.
    """)
    return


@app.cell(hide_code=True)
def _(
    a_proj,
    a_refl,
    d_shear,
    kshear,
    lam,
    mu,
    project,
    reflect,
    rotate,
    scale_dir,
    scale_uniform,
    shear,
    theta,
):
    OPERATORS = {
        f"uniform scale ×{lam:.1f}": lambda z: scale_uniform(z, lam),
        f"dir. scale ×{mu:.1f} along x": lambda z: scale_dir(z, [1.0, 0.0], mu),
        "projection onto a": lambda z: project(z, a_proj),
        "rotation by θ": lambda z: rotate(z, theta),
        f"shear k={kshear:.1f}": lambda z: shear(z, d_shear, kshear),
        "reflection in a": lambda z: reflect(z, a_refl),
    }
    return (OPERATORS,)


@app.cell(hide_code=True)
def _(OPERATORS, mo, np):
    def is_linear(f, trials=400, seed=1):
        rng = np.random.default_rng(seed)
        add_ok = hom_ok = True
        for _ in range(trials):
            p, q, c = rng.normal(size=2), rng.normal(size=2), rng.normal()
            add_ok &= np.allclose(f(p + q), f(p) + f(q))
            hom_ok &= np.allclose(f(c * p), c * f(p))
        return add_ok, hom_ok, np.allclose(f(np.zeros(2)), np.zeros(2))

    def _row(name, f):
        add_ok, hom_ok, origin_ok = is_linear(f)
        tick = {True: "✅", False: "❌"}
        return (
            f"| {name} | {tick[bool(add_ok)]} | {tick[bool(hom_ok)]} "
            f"| {tick[bool(origin_ok)]} |"
        )

    mo.md(
        "\n".join(
            [
                "| operator | additivity | homogeneity | fixes origin |",
                "|---|---|---|---|",
            ]
            + [_row(n, f) for n, f in OPERATORS.items()]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Composition, and why the order matters

    Operators are maps, so $(g\circ f)(v) = g(f(v))$ is again linear. But reversing the
    order generally changes the result. Pick any two below and compare the two pictures:
    **rotate-then-shear is not shear-then-rotate.**

    The pairs that *do* commute are worth hunting for. Two rotations always commute. A
    uniform scaling commutes with everything, because it is just multiplication by a
    number. Anything else is a coincidence of the particular parameters — drag them and
    watch the tick flip.
    """)
    return


@app.cell
def _(OPERATORS, mo):
    first = mo.ui.dropdown(
        options=list(OPERATORS.keys()), value="rotation by θ", label="apply first"
    )
    second = mo.ui.dropdown(
        options=list(OPERATORS.keys()),
        value=[k for k in OPERATORS if k.startswith("shear")][0],
        label="then apply",
    )
    mo.hstack([first, second], justify="start", gap=1.0)
    return first, second


@app.cell
def _(BLUE, GREEN, OPERATORS, check, first, mo, np, operator_view, second):
    _f = OPERATORS[first.value]
    _g = OPERATORS[second.value]

    def _fg(z):  # first f, then g
        return _g(_f(z))

    def _gf(z):  # the other order
        return _f(_g(z))

    _probe = np.array([[1.0, 0.3], [0.2, 1.7], [-1.1, 0.5]])
    _commutes = np.allclose(_fg(_probe), _gf(_probe))

    mo.vstack(
        [
            mo.hstack(
                [
                    operator_view(_fg, BLUE, width=360),
                    operator_view(_gf, GREEN, width=360),
                ],
                widths=[1, 1],
                gap=1.0,
            ),
            mo.md(
                rf"""
    **Left (blue):** {first.value}, *then* {second.value}.
    **Right (green):** {second.value}, *then* {first.value}.

    {check(r"the composite is still linear — composition never leaves the category", np.allclose(_fg(2 * _probe), 2 * _fg(_probe)))}
    {check(rf"**these two orders agree** — the operators commute", _commutes)}

    {"The two pictures are identical, so these particular operators commute." if _commutes else "The two pictures differ, so **order matters**: $g \\circ f \\neq f \\circ g$."}

    Try *uniform scale* against anything — always green, because scaling by a number
    commutes with every linear map. Try *rotation* against *shear* — always red. Matrix
    multiplication being non-commutative (notebook 5) is not an algebraic quirk; it is this
    geometric fact, written down.
    """
            ),
        ],
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Every operator was defined by **what it does to arrows** — using only lengths,
    perpendiculars, and the quarter-turn $J$ — never by a matrix:

    | operator | coordinate-free definition | preserves | defining property | nb-4 determinant |
    |---|---|---|---|---|
    | uniform scale | $S_\lambda(v)=\lambda v$ | shape (angles) | commutes with everything | $\lambda^2$ |
    | directional scale | $v+(\lambda-1)\langle v,\hat a\rangle\hat a$ | the $\perp$ direction | — | $\lambda$ |
    | projection | $\frac{\langle v,a\rangle}{\langle a,a\rangle}a$ | the line through $a$ | idempotent, $P^2=P$ | $0$ |
    | rotation | $\cos\theta\,v+\sin\theta\,Jv$ | lengths, angles, orientation | orthogonal, $\det=+1$ | $+1$ |
    | shear | $v+k\langle v,n\rangle d$ | the shear line, **area** | unipotent, $(H-I)^2=0$ | $+1$ |
    | reflection | $2P_a(v)-v$ | lengths; **reverses** orientation | involution, $F^2=I$ | $-1$ |

    - Metric operators (projection, rotation, reflection) are built from the **dot
      product**; shear and scaling are more primitive and need no metric at all.
    - The quarter-turn satisfies $J^2=-I$, and $R_\theta=\cos\theta\,I+\sin\theta\,J$ — the
      rotation operator is already "$e^{i\theta}$" in disguise.
    - Operators **compose**, and composition is generally **non-commutative**.
    - Notice the determinant column is the only one that needed a number. Every other
      column is a statement about behaviour — which is exactly the coordinate-free claim.

    **Next (notebook 3):** the two bilinear measuring tools, properly — the **dot product**
    (symmetric: lengths and angles) and the **wedge product** (antisymmetric: signed area)
    — shown to be the symmetric and antisymmetric halves of "multiplying two vectors."
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
