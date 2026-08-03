# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.16",
#     "mathviz",
#     "wigglystuff>=0.5.23",
# ]
#
# [tool.uv.sources]
# mathviz = { path = "../mathviz", editable = true }
# ///

# Geometric Linear Algebra 1 -- Linearity, as a *reactive* marimo notebook.
# A translation of 01_Linearity_Lines_to_Lines.py in which the linear map itself is
# draggable: the grid deforms under your mouse, and the theorem's claims re-verify on
# every frame.
#
#   uv run marimo edit GeometricLinearAlgebra/01_Linearity_Lines_to_Lines_marimo.py
#
# Unlike the 00 companion, this draws with `mathviz` -- the same Plane/Space3D scenes as
# the paired jupytext notebook -- rather than re-implementing the plotting inline. Two ways
# to run it, both fine:
#
#   * plain `marimo edit` (above), using the repo environment, where `marimo` and `mathviz`
#     are already dependencies of the root pyproject.toml. This is the tested path.
#   * `marimo edit --sandbox`, using the PEP 723 header above, which pulls mathviz from the
#     sibling checkout via [tool.uv.sources]. The header must stay at the TOP of the file:
#     marimo only reads inline script metadata before `import marimo`, so the same block
#     placed inside a cell is inert.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 1 — Linearity: why lines go to lines

    We start with the only two things a vector can do — **add** and **scale** — and from
    those two operations alone extract the single most important idea in the subject:
    **linearity**. A map $T$ is linear when it respects both:

    $$T(u + v) = T(u) + T(v) \qquad T(c\,v) = c\,T(v)$$

    The headline geometric fact: **a linear map sends straight lines to straight lines,
    sends parallel lines to parallel lines, and fixes the origin.** That is what makes
    linear algebra *the geometry of flat space*.

    /// tip | What the marimo version adds
    The jupytext original fixes one map $T$ and draws its deformed grid once. Here **the
    map is the input**: drag its four numbers and the grid follows, while the
    collinearity, parallelism and origin-fixing checks re-run underneath on every frame.
    The point of the theorem is that those ticks *never* go red, however hard you drag —
    linearity is not a property of the particular numbers.
    ///

    /// note | A note on coordinates
    We are doing geometry, so a vector means "an arrow / a displacement," not "a list of
    numbers." We have *no basis yet* — coordinates and matrices are not introduced until
    notebook 5. The pairs $(x, y)$ below are **plotting coordinates for the renderer**;
    the statements we make are about the arrows themselves.
    ///
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
def _(io, mo, mv, np):
    # The palette is mathviz's, which is the same set of hex constants the whole series
    # uses (see CLAUDE.md) -- no need to redeclare them per notebook any more.
    BLUE, ORANGE, GREEN = mv.BLUE, mv.ORANGE, mv.GREEN
    RED, PURPLE, GREY, FAINT = mv.RED, mv.PURPLE, mv.GREY, mv.FAINT

    def png(scene, width=430):
        """A mathviz scene -> a marimo image.

        `scene.display()` is the Jupyter path: it hands the figure to `IPython.display`
        and returns None, which marimo would render as nothing. `scene.save(buf)` takes
        the same figure through `savefig` into a BytesIO and hands the bytes back, so the
        one adapter works for both `mv.Plane` (matplotlib) and `mv.Space3D` (mplot3d).
        """
        buf = io.BytesIO()
        scene.save(buf)
        return mo.image(buf.getvalue(), width=width)

    def check(claim, ok):
        """A one-line verdict, replacing the original's `print('...', bool)` lines."""
        return f"- {'✅' if ok else '❌'} {claim}"

    def grid_lines(n=9, lim=2.0, samples=30):
        """A square grid of straight lines, as a list of (samples, 2) polylines."""
        ticks = np.linspace(-lim, lim, n)
        s = np.linspace(-lim, lim, samples)
        lines = []
        for tk in ticks:
            lines.append(np.stack([s, np.full_like(s, tk)], axis=1))  # horizontal
            lines.append(np.stack([np.full_like(s, tk), s], axis=1))  # vertical
        return lines

    def deformed(func, color, lim=2.0, extent=None):
        """The original grid (faint) overlaid with its image under `func` (bold).

        The two origin markers are the whole point of the picture: grey is where the
        origin was, green is where `func` sent it. For a linear map the green dot sits
        exactly on the grey one and you never see it move.
        """
        p = mv.Plane(extent=extent or 2.4 * lim)
        for ln in grid_lines(lim=lim):
            p.curve(ln, color=FAINT, width=1.1)
            p.curve(func(ln), color=color, width=2.0)
        p.points(np.atleast_2d([0.0, 0.0]), color=GREY, size=7.0)
        p.points(np.atleast_2d(func(np.array([0.0, 0.0]))), color=GREEN, size=12.0)
        return p

    return BLUE, FAINT, GREEN, ORANGE, PURPLE, RED, check, deformed, png


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
    ## 1. A vector is an arrow: the two operations

    Before any map can be *linear*, we need the two operations linearity is about. A
    **vector** is a displacement — an arrow with a length and a direction, free to slide
    anywhere (its tail is not pinned). There are exactly two things we can do with arrows:

    1. **Add** them, tip-to-tail: $u + v$ is "walk along $u$, then along $v$."
    2. **Scale** them by a number: $c\,v$ stretches $v$ by $c$ (and flips it if $c<0$).

    A **linear combination** $a\,u + b\,v$ is the result of doing both. The whole of flat
    geometry is generated by these moves, so a map that *commutes with both* preserves the
    geometry. Drag the four numbers and watch the parallelogram follow.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    uv = mo.ui.anywidget(
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
                    "value": 0.6,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "u, second component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "v1": {
                    "value": 0.7,
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
    uv
    return (uv,)


@app.cell(hide_code=True)
def _(BLUE, FAINT, GREEN, ORANGE, PURPLE, check, mo, mv, np, png, uv):
    # Reading `uv.values` here -- in a *different* cell from the one that defines the
    # widget, which marimo requires -- is what wires this whole cell to the mouse.
    u = np.array([uv.values["u1"], uv.values["u2"]])
    v = np.array([uv.values["v1"], uv.values["v2"]])

    _add = mv.Plane(extent=3.5)
    _add.vector(u, origin=[0, 0], color=BLUE, label="u")
    _add.vector(v, origin=u, color=ORANGE, label="v")
    _add.vector(u + v, origin=[0, 0], color=GREEN, label="u+v")
    _add.vector(v, origin=[0, 0], color=FAINT)  # faint side of the parallelogram
    _add.vector(u, origin=v, color=FAINT)  # closing side

    _scale = mv.Plane(extent=3.5)
    for _c, _col, _lab in [(1.5, GREEN, "1.5·u"), (1.0, BLUE, "u"), (-0.8, PURPLE, "-0.8·u")]:
        _scale.vector(_c * u, origin=[0, 0], color=_col, label=_lab)

    mo.vstack(
        [
            mo.hstack([png(_add), png(_scale)], widths=[1, 1], gap=1.0),
            mo.md(
                rf"""
    **Left — addition, tip-to-tail.** $u + v = ({u[0] + v[0]:.1f}, {u[1] + v[1]:.1f})$ is
    the diagonal of the parallelogram the two arrows span. **Right — scaling.** Every
    multiple of $u$ lands on one line through the origin: stretched ($c>1$), shrunk
    ($0<c<1$), or flipped ($c<0$).

    {check("addition is commutative, $u + v = v + u$", np.allclose(u + v, v + u))}
    {check("scaling distributes, $2(u+v) = 2u + 2v$", np.allclose(2 * (u + v), 2 * u + 2 * v))}

    Notice we are only *adding* and *scaling* — no other operation on arrows is allowed
    in a vector space.
    """
            ),
        ],
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. What "linear" means

    A map $T$ takes vectors to vectors. It is **linear** if it does not care whether you
    combine *before* or *after* applying it:

    $$T(a\,u + b\,v) = a\,T(u) + b\,T(v).$$

    That single equation packages both required properties (set $a=b=1$ for additivity;
    set $b=0$ for homogeneity). Two consequences fall straight out:

    - **The origin is fixed.** Put $c=0$ in homogeneity: $T(0) = T(0\cdot v) = 0\,T(v) = 0$.
      A linear map can never move the origin.
    - **Linear combinations are preserved**, by induction:
      $T\!\big(\sum_i c_i v_i\big) = \sum_i c_i\,T(v_i)$.

    To *experiment* we need concrete maps. The cleanest way to manufacture one is to pick
    where two reference arrows go and extend "by linearity."
    """)
    return


@app.cell
def _(np):
    class LinearMap:
        """A linear operator on the plane, stored by its action. NOT yet 'a matrix with a
        basis' -- that interpretation waits until notebook 5. Here it is simply a callable
        that we have *verified* satisfies additivity and homogeneity."""

        def __init__(self, action):
            self._A = np.asarray(action, float)  # 2x2 bookkeeping array

        def __call__(self, w):
            w = np.asarray(w, float)
            return w @ self._A.T  # a single vector, or a stack of row-vectors

    def cross2(a, b):
        """Scalar 2D cross product -- zero exactly when a, b are parallel."""
        return a[0] * b[1] - a[1] * b[0]

    def collinear(A, B, C, tol=1e-9):
        return abs(cross2(B - A, C - A)) < tol

    return LinearMap, collinear, cross2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The 1D case: a linear map of the line is "multiply by a number"

    Strip away the second dimension and linearity becomes startlingly rigid. On the line
    every vector is a scalar multiple of one unit arrow, $v = x\cdot e$, so by homogeneity
    a linear map is pinned down by what it does to that single arrow:

    $$T(x\,e) = x\,T(e) = x\,(a\,e) = (a x)\,e.$$

    **Every linear map of the line is multiplication by a constant** — a pure scaling.
    That is why "slope-intercept" lines $x\mapsto ax+b$ are *not* linear maps unless
    $b=0$: the intercept moves the origin. Drag $b$ off zero and watch the orange graph
    lift off the green origin dot — and the additivity tick flip to ❌.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    oned = mo.ui.anywidget(
        TangleLatex(
            latex=r"T(x) \;=\; \tangle{a}\,x \;+\; \tangle{b}",
            parameters={
                "a": {
                    "value": 1.7,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.1,
                    "digits": 1,
                    "label": "slope (the stretch factor)",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "b": {
                    "value": 0.8,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "intercept (0 = linear, anything else = affine)",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    oned
    return (oned,)


@app.cell(hide_code=True)
def _(BLUE, GREEN, ORANGE, check, mo, mv, np, oned, png):
    _a, _b = oned.values["a"], oned.values["b"]

    def _lin1d(x):
        return _a * x

    def _aff1d(x):
        return _a * x + _b

    _pp, _qq, _cc = 0.9, -1.3, 2.4
    _xx = np.linspace(-2, 2, 50)

    _p = mv.Plane(extent=4.0)
    _p.curve(np.stack([_xx, _lin1d(_xx)], axis=1), color=BLUE, width=3.0)
    _p.curve(np.stack([_xx, _aff1d(_xx)], axis=1), color=ORANGE, width=3.0)
    _p.points(np.atleast_2d([0.0, 0.0]), color=GREEN, size=10.0)

    mo.hstack(
        [
            png(_p),
            mo.md(
                rf"""
    Blue is the **linear** part $x\mapsto {_a:.1f}x$; orange is the full **affine** map
    $x\mapsto {_a:.1f}x + {_b:.1f}$. The green dot is the origin — it lies on the blue
    graph always, and on the orange graph only when the intercept is zero.

    **LINEAR** $x \mapsto {_a:.1f}x$
    {check("additivity, $T(p+q)=T(p)+T(q)$", np.allclose(_lin1d(_pp + _qq), _lin1d(_pp) + _lin1d(_qq)))}
    {check("homogeneity, $T(c\\,p)=c\\,T(p)$", np.allclose(_lin1d(_cc * _pp), _cc * _lin1d(_pp)))}

    **AFFINE** $x \mapsto {_a:.1f}x + {_b:.1f}$
    {check("additivity, $T(p+q)=T(p)+T(q)$", np.allclose(_aff1d(_pp + _qq), _aff1d(_pp) + _aff1d(_qq)))}
    {check("fixes the origin, $T(0)=0$", np.allclose(_aff1d(0.0), 0.0))}

    The affine map fails both the moment $b \neq 0$: the intercept gets counted *twice* on
    the left of $T(p+q)$ and once on the right.
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
    ## 4. The headline theorem: lines go to lines

    A straight line is the set of points you reach by starting at $p_0$ and walking along
    a fixed direction $d$:

    $$\ell(t) = p_0 + t\,d, \qquad t \in \mathbb{R}.$$

    Apply a linear map $T$ and use linearity *directly*:

    $$T(\ell(t)) = T(p_0 + t\,d) = T(p_0) + t\,T(d).$$

    The right-hand side is **again the parametric equation of a line** — it starts at
    $T(p_0)$ and runs along $T(d)$. So the image of a line is a line. (The one degenerate
    case: if $T(d)=0$ the line collapses to a point. Lines never *bend* — at worst they
    are crushed.) Two corollaries come free from the same computation: **parallel lines
    stay parallel**, because parallel lines share $d$ and their images share $T(d)$; and
    **equally-spaced points stay equally spaced**, because the parameter $t$ rides along
    untouched, so midpoints map to midpoints.

    Drag the four entries below. The faint grid is the original; the bold one is its
    image. However you distort it, the lines stay straight.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    mat = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"T \;:\; e_1 \mapsto \begin{bmatrix} \tangle{a} \\[2pt] \tangle{c} \end{bmatrix},"
                r"\quad e_2 \mapsto \begin{bmatrix} \tangle{b} \\[2pt] \tangle{d} \end{bmatrix}"
            ),
            parameters={
                "a": {
                    "value": 1.0,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "where e1 goes, first component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "c": {
                    "value": 0.5,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "where e1 goes, second component",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "b": {
                    "value": -0.4,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "where e2 goes, first component",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
                "d": {
                    "value": 1.2,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "where e2 goes, second component",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    mat
    return (mat,)


@app.cell(hide_code=True)
def _(BLUE, LinearMap, check, collinear, cross2, deformed, mat, mo, np, png):
    T = LinearMap(
        [
            [mat.values["a"], mat.values["b"]],
            [mat.values["c"], mat.values["d"]],
        ]
    )

    # Three points on one line, at three unequal parameter values.
    _p0 = np.array([-1.3, 0.4])
    _d = np.array([0.8, 1.1])
    A, B, C = _p0, _p0 + 0.37 * _d, _p0 + 0.91 * _d
    _M = 0.5 * (A + C)

    # Two *different* lines sharing the direction d -- i.e. a parallel pair. Their image
    # directions are computed from sample points rather than assumed, so the check is a
    # real one: T(p0 + d) - T(p0) versus T(q0 + d) - T(q0).
    _q0 = np.array([0.6, -1.1])
    _dir1 = T(_p0 + _d) - T(_p0)
    _dir2 = T(_q0 + _d) - T(_q0)

    _det = np.linalg.det(T._A)

    mo.hstack(
        [
            png(deformed(T, BLUE)),
            mo.md(
                rf"""
    {check("the three sample points were collinear to begin with", collinear(A, B, C))}
    {check("**and their images still are** — the line did not bend", collinear(T(A), T(B), T(C)))}
    {check("the midpoint stays the midpoint, $T(\\tfrac{{A+C}}{{2}}) = \\tfrac{{T(A)+T(C)}}{{2}}$", np.allclose(T(_M), 0.5 * (T(A) + T(C))))}
    {check("two parallel lines stay parallel — both images run along $T(d)$", abs(cross2(_dir1, _dir2)) < 1e-9)}
    {check("the origin is fixed, $T(0) = 0$ (green dot never leaves grey)", np.allclose(T(np.zeros(2)), np.zeros(2)))}

    $$\det T = {_det:.2f}$$

    The determinant is the one number that *does* change as you drag — it is the factor by
    which the map scales area, and notebook 4 is about it. Drag until it passes through
    $0$: the grid collapses onto a single line, every arrow crushed onto one direction.
    That is the degenerate case $T(d) = 0$ in the theorem, and it is the only way a line
    fails to come out a line — by becoming a point.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Affine = linear + a translation

    Add a constant $b$ *after* the linear part and you get an **affine** map. It still
    keeps every line straight and every parallel pair parallel — that is the *flatness*
    shared by linear and affine maps. The difference is the green dot: the linear map pins
    the origin, the affine map slides the whole grid off it. We focus on the linear part
    because the translation carries no *shape* information; it just relocates.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    shift = mo.ui.anywidget(
        TangleLatex(
            latex=r"x \;\mapsto\; T(x) \;+\; \begin{bmatrix} \tangle{b1} \\[2pt] \tangle{b2} \end{bmatrix}",
            parameters={
                "b1": {
                    "value": 1.3,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "translation, first component",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
                "b2": {
                    "value": -0.7,
                    "min_value": -2,
                    "max_value": 2,
                    "step": 0.1,
                    "digits": 1,
                    "label": "translation, second component",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    shift
    return (shift,)


@app.cell(hide_code=True)
def _(ORANGE, T, check, collinear, deformed, mo, np, png, shift):
    # `shift.values` is read here rather than in the cell above: marimo only re-runs a
    # cell when a value it *reads* changes, so a widget consumed in its own defining cell
    # would render once and then sit frozen under the mouse.
    def affine(w):
        return T(w) + np.array([shift.values["b1"], shift.values["b2"]])

    _p0 = np.array([-1.3, 0.4])
    _d = np.array([0.8, 1.1])
    _A, _B, _C = _p0, _p0 + 0.37 * _d, _p0 + 0.91 * _d
    _u, _v = np.array([1.0, 0.5]), np.array([-0.3, 1.4])

    mo.hstack(
        [
            png(deformed(affine, ORANGE)),
            mo.md(
                rf"""
    {check("still flat: collinear points stay collinear", collinear(affine(_A), affine(_B), affine(_C)))}
    {check("fixes the origin, $T(0) = 0$ — green sits on grey", np.allclose(affine(np.zeros(2)), np.zeros(2)))}
    {check("additivity, $T(u+v) = T(u) + T(v)$", np.allclose(affine(_u + _v), affine(_u) + affine(_v)))}

    The first tick stays green no matter where you drag — affine maps are flat too. The
    other two fail together, and for one reason: the translation $b$ is added once on the
    left of $T(u+v)$ but *twice* on the right. Set both components back to $0$ and all
    three turn green — an affine map with no translation **is** a linear map. That is the
    entire difference.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (affine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What breaks linearity: bending the grid

    To appreciate that "lines → lines" is a real constraint and not automatic, watch a
    **nonlinear** map. This one nudges each point upward by an amount proportional to
    $x^2$. It is perfectly smooth and continuous — but it is not linear, and the symptom
    is unmistakable: the straight grid lines come out **curved**.

    Drag the curvature to $0$ to recover the identity, and the parabolas snap back to
    straight lines.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    curve = mo.ui.anywidget(
        TangleLatex(
            latex=r"(x, y) \;\mapsto\; \big(x,\; y + \tangle{q}\,x^2\big)",
            parameters={
                "q": {
                    "value": 0.35,
                    "min_value": -1,
                    "max_value": 1,
                    "step": 0.05,
                    "digits": 2,
                    "label": "curvature (0 = the identity map)",
                    "color": {"light": "#a03050", "dark": "#ff8fa8"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    curve
    return (curve,)


@app.cell(hide_code=True)
def _(RED, check, collinear, curve, deformed, mo, np, png):
    def bend(w):
        w = np.asarray(w, float)
        x, y = w[..., 0], w[..., 1]
        return np.stack([x, y + curve.values["q"] * x**2], axis=-1)

    _p0 = np.array([-1.3, 0.4])
    _d = np.array([0.8, 1.1])
    _A, _B, _C = _p0, _p0 + 0.37 * _d, _p0 + 0.91 * _d

    mo.hstack(
        [
            png(deformed(bend, RED)),
            mo.md(
                rf"""
    {check("the three sample points were collinear to begin with", collinear(_A, _B, _C))}
    {check("their images are still collinear — otherwise, **the line got bent**", collinear(bend(_A), bend(_B), bend(_C)))}
    {check("it does fix the origin, $T(0)=0$ — necessary for linearity, but not sufficient", np.allclose(bend(np.zeros(2)), np.zeros(2)))}

    Note the third tick. `bend` passes the origin test that the affine map *failed*, and
    is still not linear. Fixing the origin is a **consequence** of linearity, never a
    substitute for it — the $x^2$ term breaks additivity directly, because
    $(p+q)^2 \neq p^2 + q^2$.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (bend,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The two laws, tested head-to-head

    Finally, the definition itself. We throw random vectors and random scalars at each map
    and check additivity and homogeneity. This is the *operational* meaning of linearity —
    no pictures required. The table re-runs against whatever you have dragged the three
    maps into, so the linear row stays all-green while the other two stay broken.
    """)
    return


@app.cell
def _(T, affine, bend, mo, np):
    def check_linearity(func, trials=500, seed=0):
        rng = np.random.default_rng(seed)
        add_ok = hom_ok = True
        for _ in range(trials):
            u_, v_ = rng.normal(size=2), rng.normal(size=2)
            c_ = rng.normal()
            add_ok &= np.allclose(func(u_ + v_), func(u_) + func(v_))
            hom_ok &= np.allclose(func(c_ * u_), c_ * func(u_))
        origin_ok = np.allclose(func(np.zeros(2)), np.zeros(2))
        return add_ok, hom_ok, origin_ok

    def _row(name, func):
        add_ok, hom_ok, origin_ok = check_linearity(func)
        tick = {True: "✅", False: "❌"}
        return (
            f"| {name} | {tick[bool(add_ok)]} | {tick[bool(hom_ok)]} "
            f"| {tick[bool(origin_ok)]} | {tick[bool(add_ok and hom_ok)]} |"
        )

    mo.md(
        "\n".join(
            [
                "| map | additivity | homogeneity | fixes origin | **linear?** |",
                "|---|---|---|---|---|",
                _row("`T` &mdash; linear", T),
                _row("`T(x) + b` &mdash; affine", affine),
                _row("`bend` &mdash; nonlinear", bend),
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - A **vector** is an arrow you can **add** (tip-to-tail) and **scale**. Those two
      operations are the entire raw material of linear algebra.
    - A map is **linear** iff $T(au + bv) = a\,T(u) + b\,T(v)$ — additive *and*
      homogeneous.
    - Consequences, proved straight from the definition:
        - $T(0) = 0$ — **the origin is fixed**.
        - **Lines go to lines**: $T(p_0 + t\,d) = T(p_0) + t\,T(d)$ is again a line (or,
          if $T(d)=0$, a point).
        - **Parallel lines stay parallel**, and **midpoints map to midpoints**.
    - In **1D**, every linear map is multiplication by a constant — a pure scaling.
      Slope-intercept "lines" $ax+b$ are *affine*, not linear, the moment $b \neq 0$.
    - **Affine = linear + translation.** The translation only relocates; all the *shape*
      of the map lives in its linear part.
    - The one thing that *did* vary as you dragged was $\det T$, the area-scaling factor.
      Hold that thought — it is notebook 4.

    **Next (notebook 2):** we stop *manufacturing* linear maps and start **defining the
    important ones geometrically** — projection onto a vector, rotation, scaling, shear —
    entirely without coordinates, using only lengths and the right-angle relationship
    between arrows.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
