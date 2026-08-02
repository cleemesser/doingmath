# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mathviz",
#     "wigglystuff>=0.5.23",
# ]
#
# [tool.uv.sources]
# mathviz = { path = "../mathviz", editable = true }
# ///

# Geometric Linear Algebra 3 -- dot and wedge, as a *reactive* marimo notebook.
# A translation of 03_Dot_and_Wedge.py in which one draggable pair (u, v) drives the whole
# notebook: the dot product, the signed area, the Pythagorean identity that locks them
# together, and -- in 3D -- the bivector and its cross-product stand-in.
#
#   uv run marimo edit GeometricLinearAlgebra/03_Dot_and_Wedge_marimo.py
#
# Draws with `mathviz`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 3 — dot and wedge: lengths, angles, and signed area

    Two arrows can be multiplied in exactly two structurally different ways, and between
    them they say everything there is to say about the pair:

    | | dot product | wedge product |
    |---|---|---|
    | formula | $\lvert u\rvert\lvert v\rvert\cos\theta$ | $\lvert u\rvert\lvert v\rvert\sin\theta$ |
    | symmetry | **symmetric**, $\langle u,v\rangle=\langle v,u\rangle$ | **antisymmetric**, $u\wedge v=-v\wedge u$ |
    | degenerate when | the arrows are perpendicular | the arrows are parallel |
    | measures | alignment — length, angle | spread — **signed area** |

    They are not rivals; they are the symmetric and antisymmetric halves of one product,
    locked together by a Pythagorean identity we prove below. Everything on this page is
    driven by a single draggable pair $(u, v)$ — drag it until the dot product vanishes,
    then until the wedge does, and watch which picture degenerates each time.
    """)
    return


@app.cell
def _():
    import io

    import marimo as mo
    import numpy as np

    import mathviz as mv  # shared plane-viz library (see ../mathviz)
    from wigglystuff import TangleLatex

    return TangleLatex, io, mo, mv, np


@app.cell
def _(io, mo, mv):
    BLUE, ORANGE, GREEN = mv.BLUE, mv.ORANGE, mv.GREEN
    RED, PURPLE, GREY, FAINT = mv.RED, mv.PURPLE, mv.GREY, mv.FAINT

    def png(scene, width=430):
        """A mathviz scene -> a marimo image (see 01_..._marimo.py for why not .display())."""
        buf = io.BytesIO()
        scene.save(buf)
        return mo.image(buf.getvalue(), width=width)

    def check(claim, ok):
        return f"- {'✅' if ok else '❌'} {claim}"

    def space(bounds=3.0, **kw):
        """A 3D scene on the *matplotlib* backend.

        mathviz defaults Space3D to vedo, which renders this scene in ~0.97s against
        mpl's ~0.28s. A reactive notebook re-renders on every drag, so the cheaper
        backend is the right default here -- and it keeps 2D and 3D on one renderer.
        """
        return mv.Space3D(bounds=bounds, backend="mpl", **kw)

    return BLUE, FAINT, GREEN, GREY, ORANGE, PURPLE, RED, check, png, space


@app.cell
def _(np):
    def dot(u, v):
        return np.sum(np.asarray(u, float) * np.asarray(v, float), axis=-1)

    def norm(v):
        return np.sqrt(dot(v, v))

    def wedge2(u, v):
        """Signed area of the parallelogram spanned by u, v in the plane."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def cross3(u, v):
        """The 3D wedge's classical stand-in: length = area, direction perpendicular."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        return np.array(
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        )

    def polygon_area(verts):
        """Signed area via the shoelace = half-sum of consecutive wedges."""
        verts = np.asarray(verts, float)
        return 0.5 * np.sum(wedge2(verts, np.roll(verts, -1, axis=0)))

    def apply_lin(A, v):
        """Apply the 2x2 bookkeeping array A to a vector or a stack of row-vectors."""
        return np.asarray(v, float) @ np.asarray(A, float).T

    return apply_lin, cross3, dot, norm, polygon_area, wedge2


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
    ## 1. The dot product, properly

    Define $\langle u,v\rangle = \lvert u\rvert\,\lvert v\rvert\,\cos\theta$. Three
    structural facts make it the unique sensible "ruler":

    - **Symmetric**: $\langle u,v\rangle = \langle v,u\rangle$ — the angle does not care
      about order.
    - **Bilinear**: linear in each slot separately.
    - **Positive-definite**: $\langle v,v\rangle = \lvert v\rvert^2 \ge 0$, zero only for
      the zero arrow.

    From these alone come the headline results of plane trigonometry: the **law of
    cosines** (just expand $\lvert u-v\rvert^2$ by bilinearity) and the
    **Cauchy–Schwarz inequality**, which is nothing but the statement that
    $\lvert\cos\theta\rvert\le 1$.
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
                "u1": {"value": 2.0, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "u, first component",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "u2": {"value": 0.5, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "u, second component",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "v1": {"value": 0.6, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "v, first component",
                       "color": {"light": "#b45b1b", "dark": "#ffad66"}},
                "v2": {"value": 1.7, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "v, second component",
                       "color": {"light": "#b45b1b", "dark": "#ffad66"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    duo
    return (duo,)


@app.cell
def _(duo, np):
    # The one pair that drives sections 1-3. Read from the widget here so every downstream
    # cell depends on `u`/`v` rather than on the widget itself.
    u = np.array([duo.values["u1"], duo.values["u2"]])
    v = np.array([duo.values["v1"], duo.values["v2"]])
    return u, v


@app.cell(hide_code=True)
def _(BLUE, GREY, ORANGE, check, dot, mo, mv, norm, np, png, u, v):
    _theta = np.arccos(np.clip(dot(u, v) / (norm(u) * norm(v)), -1, 1))

    _p = mv.Plane(extent=3.0)
    _p.vector(u, color=BLUE, label="u")
    _p.vector(v, color=ORANGE, label="v")
    _p.segment(u, v, color=GREY, width=1.2)  # the third side, |u − v|

    mo.hstack(
        [
            png(_p),
            mo.md(
                rf"""
    $$\langle u,v\rangle = {dot(u, v):+.3f} \qquad
      \lvert u\rvert = {norm(u):.3f} \qquad
      \lvert v\rvert = {norm(v):.3f} \qquad
      \theta = {np.degrees(_theta):.1f}^\circ$$

    {check(r"symmetric, $\langle u,v\rangle = \langle v,u\rangle$", np.allclose(dot(u, v), dot(v, u)))}
    {check(r"positive-definite, $\langle u,u\rangle = \lvert u\rvert^2 > 0$", dot(u, u) > 0)}
    {check(r"**law of cosines**, $\lvert u-v\rvert^2 = \lvert u\rvert^2 + \lvert v\rvert^2 - 2\langle u,v\rangle$", np.allclose(norm(u - v) ** 2, dot(u, u) + dot(v, v) - 2 * dot(u, v)))}
    {check(r"**Cauchy–Schwarz**, $\lvert\langle u,v\rangle\rvert \le \lvert u\rvert\lvert v\rvert$", abs(dot(u, v)) <= norm(u) * norm(v) + 1e-12)}
    {check(r"…with **equality**, so $u$ and $v$ are parallel right now", np.allclose(abs(dot(u, v)), norm(u) * norm(v)))}

    The grey segment is the third side of the triangle, $\lvert u-v\rvert$ — the law of
    cosines is the statement relating it to the other two sides and the angle. Drag the
    two arrows onto the same line and Cauchy–Schwarz tightens to equality; that is the
    *only* way to make the last tick green.
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
    ## 2. The wedge product: signed area

    Now the antisymmetric partner. In 2D the **wedge** $u \wedge v$ is the **signed area**
    of the parallelogram spanned by $u$ and $v$:

    $$u \wedge v = \lvert u\rvert\,\lvert v\rvert\,\sin\theta = u_x v_y - u_y v_x.$$

    Its defining properties mirror the dot's:

    - **Antisymmetric**: $u\wedge v = -\,v\wedge u$. Swapping flips the orientation, hence
      the sign. The magnitude is the area; the sign records whether $v$ is counter-clockwise
      from $u$ ($+$) or clockwise ($-$).
    - **Bilinear**: sliding one edge parallel to the other shears the parallelogram without
      changing base × height.
    - **Alternating**: $u\wedge u = 0$ — a degenerate parallelogram has no area.

    That last property is exactly the collinearity test from notebook 1: "lies on a line"
    means "spans zero area." The two pictures below are the *same* parallelogram in the two
    orders; only the sign, and so the color, differs.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, GREEN, ORANGE, RED, check, mo, mv, norm, np, png, u, v, wedge2, dot):
    _uv, _vu = wedge2(u, v), wedge2(v, u)
    _theta = np.arccos(np.clip(dot(u, v) / (norm(u) * norm(v)), -1, 1))

    def _pane(a, b, area, la, lb, ca, cb):
        p = mv.Plane(extent=3.0)
        p.parallelogram([0, 0], a, b, facecolor=(GREEN if area > 0 else RED), alpha=0.22)
        p.vector(a, color=ca, label=la)
        p.vector(b, color=cb, label=lb)
        return png(p, width=360)

    mo.vstack(
        [
            mo.hstack(
                [
                    _pane(u, v, _uv, "u", "v", BLUE, ORANGE),
                    _pane(v, u, _vu, "v", "u", ORANGE, BLUE),
                ],
                widths=[1, 1],
                gap=1.0,
            ),
            mo.md(
                rf"""
    $$u \wedge v = {_uv:+.3f} \qquad\qquad v \wedge u = {_vu:+.3f}$$

    {check(r"**antisymmetric**, $u\wedge v = -\,v\wedge u$", np.allclose(_uv, -_vu))}
    {check(r"**alternating**, $u\wedge u = 0$", np.allclose(wedge2(u, u), 0.0))}
    {check(r"$u\wedge v = \lvert u\rvert\lvert v\rvert\sin\theta$", np.allclose(abs(_uv), norm(u) * norm(v) * np.sin(_theta)))}
    {check(r"the area is **positive** — $v$ is counter-clockwise from $u$", _uv > 0)}
    {check(r"$u$ and $v$ are parallel right now, so the area has collapsed to $0$", np.allclose(_uv, 0.0))}

    Green is a positive (counter-clockwise) orientation, red negative. The two panes are
    the same set of points — the same parallelogram, the same area — and the wedge still
    tells them apart, because it measures the *ordered* pair. That sign is the whole
    reason the wedge exists; an unsigned area would lose it.
    """
            ),
        ],
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Dot + wedge = the whole story

    Alignment is $\cos\theta$, spread is $\sin\theta$, and $\cos^2\theta+\sin^2\theta=1$.
    Multiply through by $\lvert u\rvert^2\lvert v\rvert^2$:

    $$\langle u,v\rangle^2 + (u\wedge v)^2 = \lvert u\rvert^2\,\lvert v\rvert^2.$$

    So the dot and the wedge are genuinely the two orthogonal *components* of the pair
    $(u,v)$: knowing both lengths plus these two numbers pins down the relative geometry
    completely, up to a rigid motion. This is the 2D shadow of the **geometric product**
    $uv = \langle u,v\rangle + u\wedge v$, in which a scalar part and an area part are
    literally added together — the idea geometric algebra is built on.
    """)
    return


@app.cell
def _(check, dot, mo, norm, np, u, v, wedge2):
    _lhs = dot(u, v) ** 2 + wedge2(u, v) ** 2
    _rhs = dot(u, u) * dot(v, v)

    # The identity is not a fact about *these* two arrows -- confirm on random pairs too.
    _rng = np.random.default_rng(2)
    _always = True
    for _ in range(5000):
        _p, _q = _rng.normal(size=2), _rng.normal(size=2)
        _always &= np.allclose(dot(_p, _q) ** 2 + wedge2(_p, _q) ** 2, dot(_p, _p) * dot(_q, _q))

    mo.md(
        rf"""
    $$\underbrace{{({dot(u, v):+.3f})^2}}_{{\langle u,v\rangle^2}}
      + \underbrace{{({wedge2(u, v):+.3f})^2}}_{{(u\wedge v)^2}}
      = {_lhs:.4f}
      \qquad
      \lvert u\rvert^2\lvert v\rvert^2 = {norm(u) ** 2:.3f} \times {norm(v) ** 2:.3f}
      = {_rhs:.4f}$$

    {check("the identity holds for the pair you have dragged to", np.allclose(_lhs, _rhs))}
    {check("…and for 5000 random pairs besides", _always)}

    Drag $u$ and $v$ around and watch the two squared terms trade off against each other
    while their sum sits perfectly still. Perpendicular arrows put everything in the wedge
    term; parallel arrows put everything in the dot term. Every other configuration splits
    it — and the split is exactly $\cos^2 + \sin^2$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Areas of polygons — the shoelace formula *is* a sum of wedges

    Because the wedge is signed and bilinear, the area of any polygon is half the sum of
    the wedges of consecutive vertex vectors — the **shoelace formula**, now revealed as
    "add up signed triangle areas and let the outside cancel":

    $$\text{Area} = \tfrac12\sum_i v_i \wedge v_{i+1}.$$

    The shaded fan shows the cancellation: triangles swept counter-clockwise from the first
    vertex count positive, clockwise ones negative, and the parts sticking outside the
    polygon are subtracted off exactly. Drag one vertex — including *through* an edge, so
    the polygon self-intersects — and watch the signed area stay coherent.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    vert = mo.ui.anywidget(
        TangleLatex(
            latex=r"v_3 = \begin{bmatrix} \tangle{x} \\[2pt] \tangle{y} \end{bmatrix}",
            parameters={
                "x": {"value": 2.6, "min_value": -3, "max_value": 3, "step": 0.1,
                      "digits": 1, "label": "third vertex, x",
                      "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "y": {"value": 1.8, "min_value": -3, "max_value": 3, "step": 0.1,
                      "digits": 1, "label": "third vertex, y",
                      "color": {"light": "#147a68", "dark": "#5ed5bd"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    reverse = mo.ui.switch(label="traverse the vertices in reverse order")
    mo.hstack([vert, reverse], justify="start", align="center", gap=2.0)
    return reverse, vert


@app.cell(hide_code=True)
def _(BLUE, GREEN, ORANGE, RED, check, mo, mv, np, png, polygon_area, reverse, vert):
    poly = np.array(
        [[0.0, 0.0], [2.0, 0.3], [vert.values["x"], vert.values["y"]], [1.0, 2.6], [-0.5, 1.4]]
    )
    if reverse.value:
        poly = poly[::-1]

    _area = polygon_area(poly)
    _p = mv.Plane(extent=3.2)
    # The fan of signed triangles from vertex 0 -- this is the shoelace sum, drawn.
    for _i in range(1, len(poly) - 1):
        _tri = np.array([poly[0], poly[_i], poly[_i + 1]])
        _p.polygon(_tri, facecolor=(GREEN if polygon_area(_tri) > 0 else RED), alpha=0.16)
    _p.curve(np.vstack([poly, poly[0]]), color=BLUE, width=3.0, closed=False)
    _p.points(poly, color=ORANGE, size=9.0)

    # Cross-check against a shape whose area is unarguable: a 3-4-5-ish right triangle.
    _tri345 = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 2.0]])

    mo.hstack(
        [
            png(_p),
            mo.md(
                rf"""
    $$\text{{signed area}} = \tfrac12\sum_i v_i \wedge v_{{i+1}} = {_area:+.4f}$$

    {check(rf"the traversal is **counter-clockwise** (positive area)", _area > 0)}
    {check(r"shoelace agrees with $\tfrac12\,\text{base}\times\text{height}$ on a test triangle", np.allclose(polygon_area(_tri345), 0.5 * 3.0 * 2.0))}
    {check(r"reversing the vertex order negates the area", np.allclose(polygon_area(poly), -polygon_area(poly[::-1])))}
    {check(r"the answer does not depend on where you start the loop", np.allclose(polygon_area(poly), polygon_area(np.roll(poly, 2, axis=0))))}

    Flip the switch and every number changes sign while the picture stays put — the
    polygon has no intrinsic orientation, the *traversal* does. And that last tick is
    quietly important: the shoelace sum is a loop, so it cannot depend on which vertex you
    call first. Both facts are inherited straight from the wedge's antisymmetry.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (poly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Into 3D: the wedge as an oriented area element

    In space the parallelogram spanned by $u$ and $v$ still has an area, but now it also
    has an *orientation in 3D* — it tilts. The wedge becomes a **bivector**: an oriented
    patch of plane whose magnitude is the area and whose attitude is the plane it lies in.

    The classical stand-in for that bivector is the **cross product** $u\times v$ — the
    vector perpendicular to the patch, whose length is the area and whose direction
    (right-hand rule) encodes the orientation:

    $$\lvert u\times v\rvert = \lvert u\rvert\,\lvert v\rvert\,\sin\theta = \text{area},
      \qquad u\times v \perp u,\ v.$$

    That substitution is a happy accident of three dimensions: a plane in 3D has exactly
    one perpendicular direction, so a 2-dimensional patch can be labelled by a
    1-dimensional arrow. In 4D it fails — there is a whole plane of perpendiculars — and
    the bivector is what survives. Notebook 8 builds it properly.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    three = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"v = \begin{bmatrix} \tangle{a} \\[2pt] \tangle{b} \\[2pt] \tangle{c} \end{bmatrix}"
                r"\qquad u = \begin{bmatrix} 2.0 \\[2pt] 0.4 \\[2pt] 0.3 \end{bmatrix}"
            ),
            parameters={
                "a": {"value": 0.5, "min_value": -2.5, "max_value": 2.5, "step": 0.1,
                      "digits": 1, "label": "v, x-component",
                      "color": {"light": "#b45b1b", "dark": "#ffad66"}},
                "b": {"value": 1.8, "min_value": -2.5, "max_value": 2.5, "step": 0.1,
                      "digits": 1, "label": "v, y-component",
                      "color": {"light": "#b45b1b", "dark": "#ffad66"}},
                "c": {"value": 0.2, "min_value": -2.5, "max_value": 2.5, "step": 0.1,
                      "digits": 1, "label": "v, z-component",
                      "color": {"light": "#b45b1b", "dark": "#ffad66"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    three
    return (three,)


@app.cell(hide_code=True)
def _(BLUE, GREEN, ORANGE, check, cross3, dot, mo, norm, np, png, space, three):
    u3 = np.array([2.0, 0.4, 0.3])
    v3 = np.array([three.values["a"], three.values["b"], three.values["c"]])
    n3 = cross3(u3, v3)
    _area = float(np.linalg.norm(n3))

    _s = space(bounds=3.0)
    _s.parallelogram([0, 0, 0], u3, v3, facecolor=BLUE, alpha=0.3)
    _s.arrow([0, 0, 0], u3, color=BLUE)
    _s.arrow([0, 0, 0], v3, color=ORANGE)
    _s.arrow([0, 0, 0], n3, color=GREEN)  # the area-vector, normal to the patch

    mo.hstack(
        [
            png(_s, width=430),
            mo.md(
                rf"""
    $$u\times v = ({n3[0]:+.2f},\; {n3[1]:+.2f},\; {n3[2]:+.2f})
      \qquad \lvert u\times v\rvert = {_area:.4f} = \text{{area}}$$

    {check(r"$u\times v \perp u$", np.allclose(dot(n3, u3), 0))}
    {check(r"$u\times v \perp v$", np.allclose(dot(n3, v3), 0))}
    {check(r"$\lvert u\times v\rvert^2 + \langle u,v\rangle^2 = \lvert u\rvert^2\lvert v\rvert^2$ — the same identity, in space", np.allclose(_area ** 2 + dot(u3, v3) ** 2, dot(u3, u3) * dot(v3, v3)))}
    {check(r"antisymmetric in 3D too, $u\times v = -\,v\times u$", np.allclose(n3, -cross3(v3, u3)))}
    {check(r"$u$ and $v$ are parallel, so the patch has collapsed", np.allclose(_area, 0.0))}

    Blue and orange span the patch; green is the area-vector standing normal to it, and
    its *length* — not just its direction — is the number you want. Drag $v$ onto $u$'s
    line: the patch degenerates, the green arrow shrinks to nothing, and the whole of
    $\lvert u\rvert^2\lvert v\rvert^2 = {dot(u3, u3) * dot(v3, v3):.3f}$ moves into the dot-product term.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return n3, u3, v3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. How operators act on area — the bridge to the determinant

    Here is the punchline that notebook 4 cashes in. Take *any* linear operator $T$ and
    feed a parallelogram through it. By bilinearity and antisymmetry of the wedge, the
    image area is the original area times a factor that **does not depend on which $u,v$
    you chose**:

    $$T(u)\wedge T(v) = \big(\det T\big)\;(u\wedge v).$$

    That universal factor is the **determinant** — defined right here, coordinate-free, as
    *the number by which $T$ multiplies every signed area*. Drag the operator's entries:
    the six ratios below are computed from six unrelated random pairs, and they stay
    locked to each other. That constancy is the theorem.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    opw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"T = \begin{bmatrix} \tangle{a} & \tangle{b} \\[2pt]"
                r"\tangle{c} & \tangle{d} \end{bmatrix}"
            ),
            parameters={
                "a": {"value": 1.3, "min_value": -2, "max_value": 2, "step": 0.1,
                      "digits": 1, "label": "T, row 1 col 1",
                      "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "b": {"value": -0.7, "min_value": -2, "max_value": 2, "step": 0.1,
                      "digits": 1, "label": "T, row 1 col 2",
                      "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "c": {"value": 0.4, "min_value": -2, "max_value": 2, "step": 0.1,
                      "digits": 1, "label": "T, row 2 col 1",
                      "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "d": {"value": 1.1, "min_value": -2, "max_value": 2, "step": 0.1,
                      "digits": 1, "label": "T, row 2 col 2",
                      "color": {"light": "#147a68", "dark": "#5ed5bd"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    opw
    return (opw,)


@app.cell(hide_code=True)
def _(BLUE, GREEN, apply_lin, check, mo, mv, np, opw, png, wedge2):
    A = np.array(
        [[opw.values["a"], opw.values["b"]], [opw.values["c"], opw.values["d"]]]
    )

    _rng = np.random.default_rng(7)
    _ratios = []
    for _ in range(6):
        _p, _q = _rng.normal(size=2), _rng.normal(size=2)
        _w = wedge2(_p, _q)
        if abs(_w) > 1e-9:
            _ratios.append(wedge2(apply_lin(A, _p), apply_lin(A, _q)) / _w)
    _det = float(np.linalg.det(A))
    _constant = np.allclose(_ratios, _ratios[0])

    _e1, _e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _p2 = mv.Plane(extent=3.0)
    _p2.parallelogram([0, 0], _e1, _e2, facecolor=mv.FAINT, alpha=0.35)
    _p2.parallelogram(
        [0, 0], apply_lin(A, _e1), apply_lin(A, _e2), facecolor=BLUE, alpha=0.25
    )
    _p2.vector(apply_lin(A, _e1), color=BLUE, label="Te₁")
    _p2.vector(apply_lin(A, _e2), color=GREEN, label="Te₂")

    mo.hstack(
        [
            png(_p2),
            mo.md(
                rf"""
    ratios $\dfrac{{T(u)\wedge T(v)}}{{u\wedge v}}$ for six unrelated random pairs:

    $$[\;{", ".join(f"{r:+.6f}" for r in _ratios)}\;]$$

    {check(r"**all six agree** — the factor is a property of $T$, not of $u$ and $v$", _constant)}
    {check(rf"and that common value is $\det T = {_det:+.4f}$", np.allclose(_ratios[0], _det))}
    {check(r"$T$ preserves area ($\det T = 1$)", np.allclose(_det, 1.0))}
    {check(r"$T$ preserves **orientation** ($\det T > 0$)", _det > 0)}

    The faint square is the unit square (area $1$); the blue parallelogram is its image
    (area $\lvert\det T\rvert = {abs(_det):.4f}$). Drag the entries until $\det T$ crosses
    zero: the blue parallelogram flattens to a segment and then re-inflates *inverted* —
    the orientation tick flips, and the picture is the reason the determinant needed a
    sign in the first place.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The notebook-2 operators, as area-scaling factors

    Read off the determinant column we previewed in notebook 2 — now *derived* rather than
    asserted, each entry computed as $T(e_1)\wedge T(e_2)$.
    """)
    return


@app.cell
def _(apply_lin, mo, np, wedge2):
    _ops = {
        "rotation 36°": np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]]),
        "uniform scale 1.6": np.array([[1.6, 0.0], [0.0, 1.6]]),
        "shear k=0.8": np.array([[1.0, 0.8], [0.0, 1.0]]),
        "projection onto x": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "reflection in x": np.array([[1.0, 0.0], [0.0, -1.0]]),
    }
    _e1, _e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    def _row(name, M):
        area = wedge2(apply_lin(M, _e1), apply_lin(M, _e2)) / wedge2(_e1, _e2)
        note = {
            True: "preserves area",
            False: "changes area",
        }[bool(np.isclose(abs(area), 1.0))]
        if np.isclose(area, 0.0):
            note = "**crushes** the plane — not invertible"
        elif area < 0:
            note = "**reverses** orientation"
        return f"| {name} | {area:+.3f} | {note} |"

    mo.md(
        "\n".join(
            ["| operator | area factor = det | |", "|---|---|---|"]
            + [_row(n, M) for n, M in _ops.items()]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As promised in notebook 2: rotation and shear preserve area ($\det=+1$), uniform
    scaling multiplies it by $\lambda^2$ ($1.6^2 = 2.56$), projection crushes it to $0$,
    and reflection gives $-1$ — the minus sign being exactly the orientation flip the
    wedge was built to detect.

    ## Summary

    - The **dot product** is the *symmetric* bilinear pairing
      $\lvert u\rvert\lvert v\rvert\cos\theta$: lengths, angles, perpendicularity, the law
      of cosines, Cauchy–Schwarz.
    - The **wedge product** is the *antisymmetric* bilinear pairing
      $\lvert u\rvert\lvert v\rvert\sin\theta$: **signed area**, antisymmetric, alternating
      ($u\wedge u = 0$). Its sign is orientation.
    - They are two halves of one product, locked together by
      $\langle u,v\rangle^2 + (u\wedge v)^2 = \lvert u\rvert^2\lvert v\rvert^2$ — the
      geometric-product identity in miniature. Dragging $(u,v)$ only ever moves weight
      between the two terms; it never changes the total.
    - **Area of any polygon** = half-sum of consecutive wedges: the shoelace formula.
    - In **3D** the wedge is an oriented area element (a **bivector**), classically stood
      in for by the **cross product**: length = area, direction perpendicular.
    - **Operators scale every signed area by one universal factor**,
      $T(u)\wedge T(v) = (\det T)(u\wedge v)$ — the coordinate-free **determinant**.

    **Next (notebook 4):** the top-degree wedge as the **volume element**, the
    **determinant** as the volume-scaling factor in 2D *and* 3D, and a **coordinate-free
    trace** as the first-order rate at which an operator changes volume.
    """)
    return


if __name__ == "__main__":
    app.run()
