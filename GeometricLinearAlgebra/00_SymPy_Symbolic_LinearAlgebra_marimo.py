# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "wigglystuff>=0.5.22",
#     "sympy>=1.14",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
#     "ruff==0.16.1",
#     "duckdb==1.5.5",
#     "sqlglot==30.14.0",
#     "polars[pyarrow]==1.43.2",
#     "nbformat==5.10.4",
# ]
# ///
#
# Geometric Linear Algebra 0 -- the symbolic companion, as a *reactive* marimo notebook.
# A translation of 00_SymPy_Symbolic_LinearAlgebra_experiments.py, with the rotation
# matrix (and friends) made draggable via wigglystuff's TangleLatex widget.
#
#   uv run marimo edit --sandbox GeometricLinearAlgebra/00_SymPy_Symbolic_LinearAlgebra_marimo.py
#
# The --sandbox flag makes uv build a throwaway venv from the PEP 723 header above, so
# this file runs without touching the repo's own pyproject/uv.lock.

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 0 — the symbolic companion via `sympy`

    The numerical notebooks in this series prove their claims with `np.allclose` on one
    random sample. The SymPy companion proves them **as identities**: symbols go inside matrix expressions. We use sympy's `simplify`s to show that an equation holds for *all* inputs
    at once.

    Each section comes in two parts:

    1. **the proof for all values** — pure SymPy, symbols in the matrix, `simplify(...) == 0`;
    2. **the feel of one value** — a [`TangleLatex`](https://koaning.github.io/wigglystuff/)
       formula whose numbers you drag with the mouse, with SymPy recomputing the exact
       consequence underneath as you drag.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from sympy import Matrix, eye, symbols, cos, sin, simplify, trigsimp
    from wigglystuff import TangleLatex

    return (
        Matrix,
        TangleLatex,
        cos,
        eye,
        mo,
        np,
        plt,
        simplify,
        sin,
        sp,
        symbols,
        trigsimp,
    )


@app.cell
def _(np, plt, sp):
    # Colors carried over from the clmmathtools original (see CLAUDE.md: named hex constants,
    # reused for visual consistency). clmmathtools's vedo offscreen Plotter is replaced by matplotlib
    # here -- marimo renders a returned Figure directly, and re-renders it on every drag,
    # which is what we want for a reactive notebook.
    GREY = "#888888"
    RED = "#ef5350"
    GREEN = "#33c463"
    BLUE = "#4fc3f7"
    ORANGE = "#ffb74d"
    PURPLE = "#ce93d8"

    def plane(extent=3, size=(4.3, 4.3)):
        """A square, equal-aspect coordinate plane -- the matplotlib stand-in for Plane2D."""
        fig, ax = plt.subplots(figsize=size)
        ticks = np.arange(-extent, extent + 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, color=GREY, alpha=0.22, lw=0.8)
        ax.axhline(0, color=GREY, lw=1.0, alpha=0.55)
        ax.axvline(0, color=GREY, lw=1.0, alpha=0.55)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(labelsize=7, colors=GREY, length=0)
        fig.tight_layout()
        return fig, ax

    def arrow(ax, vec, origin=(0, 0), color=RED, label=None, alpha=1.0, lw=2.2):
        o = np.asarray(origin, float)
        e = o + np.asarray(vec, float)
        ax.annotate(
            "",
            xy=e,
            xytext=o,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=16,
            ),
        )
        if label:
            ax.text(*(e + 0.13), label, color=color, fontsize=10, alpha=alpha)

    def polygon(ax, pts, color=BLUE, lw=2.0, fill=0.13):
        p = np.asarray(pts, float)
        closed = np.vstack([p, p[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw)
        if fill:
            ax.fill(p[:, 0], p[:, 1], color=color, alpha=fill)

    def check(claim, ok):
        """A one-line verdict, replacing the original's `print('...', bool)` lines."""
        return f"- {'✅' if ok else '❌'} {claim}"

    def exact(value):
        """A dragged float -> an *exact* rational (steps are chosen so this is lossless)."""
        return sp.Rational(str(round(float(value), 6)))

    def exact_deg(degrees):
        """A dragged angle in degrees -> the exact symbolic radian measure."""
        return sp.Rational(int(round(float(degrees))), 180) * sp.pi

    return (
        BLUE,
        GREEN,
        GREY,
        ORANGE,
        PURPLE,
        RED,
        arrow,
        check,
        exact,
        exact_deg,
        plane,
        polygon,
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
    Values shown as a Greek letter reveal their number while you drag.
    ///
    """
    )
    return (theme,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The operators as exact matrices (notebook 2, symbolically)

    The geometric operators were defined coordinate-free; with a basis they become matrices.
    Start with the quarter-turn $J$ — the seed of $i^2=-1$ — and the rotation
    $R(\theta)=\cos\theta\,I+\sin\theta\,J$.
    """)
    return


@app.cell
def _(Matrix, check, cos, eye, mo, sin, sp, symbols, trigsimp):
    th, al, be = symbols("theta alpha beta", real=True)
    J = Matrix([[0, -1], [1, 0]])
    I2 = eye(2)

    R = cos(th) * I2 + sin(th) * J
    Ra = cos(al) * I2 + sin(al) * J
    Rb = cos(be) * I2 + sin(be) * J
    Rab = cos(al + be) * I2 + sin(al + be) * J

    mo.md(
        rf"""
    $$J = {sp.latex(J)}, \qquad J^2 = {sp.latex(J * J)} = -I$$

    $$R(\theta) \;=\; \cos\theta\,I + \sin\theta\,J \;=\; {sp.latex(R)}
      \qquad R^{{\top}}R \;=\; {sp.latex(trigsimp(R.T * R))}$$

    {check(r"$R^\top R = I$ — rotation is orthogonal", trigsimp(R.T * R) == I2)}
    {check(r"$\det R = 1$ — orientation and area preserved", trigsimp(R.det()) == 1)}
    {
            check(
                r"$R(\alpha)R(\beta) = R(\alpha{+}\beta)$ — the angle-sum identities for "
                r"$\sin$ and $\cos$, falling out of matrix multiplication",
                trigsimp(Ra * Rb - Rab) == sp.zeros(2),
            )
        }

    Those three ticks are proofs **for every angle at once** — symbols went into the
    matrix, and the difference simplified to the zero matrix. Now pick one angle and
    feel it.
    """
    )
    return I2, J, R, Ra, Rab, Rb, al, be, th


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    rot = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"R(\tangle{theta}^\circ) \;=\; \begin{bmatrix}"
                r"\cos \tangle{theta}^\circ & -\sin \tangle{theta}^\circ \\[2pt]"
                r"\sin \tangle{theta}^\circ & \cos \tangle{theta}^\circ"
                r"\end{bmatrix}"
            ),
            parameters={
                "theta": {
                    "value": 30,
                    "min_value": -180,
                    "max_value": 180,
                    "step": 15,  # 15deg steps keep Rational(deg,180)*pi exact
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
    rot
    return (rot,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREEN,
    GREY,
    I2,
    R,
    RED,
    arrow,
    check,
    exact_deg,
    mo,
    np,
    plane,
    polygon,
    rot,
    sp,
    th,
):
    # The reactive core: `rot.values["theta"]` is the dragged number. Reading it here
    # (in a *different* cell from the one that defines the widget -- marimo requires
    # that) wires this whole cell to the mouse.
    _deg = rot.values["theta"]
    theta_x = exact_deg(_deg)  # e.g. 30 -> pi/6, exactly
    R_x = sp.simplify(R.subs(th, theta_x))

    _fig, _ax = plane(extent=2)
    _num = np.array(sp.matrix2numpy(R_x, dtype=float))
    polygon(_ax, [[0, 0], [1, 0], [1, 1], [0, 1]], GREY, lw=1.2, fill=0.06)
    polygon(_ax, (_num @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]])).T, BLUE, lw=2.0)
    arrow(_ax, [1, 0], color=GREY, alpha=0.45)
    arrow(_ax, [0, 1], color=GREY, alpha=0.45)
    arrow(_ax, _num @ [1, 0], color=RED, label=r"$Re_1$")
    arrow(_ax, _num @ [0, 1], color=GREEN, label=r"$Re_2$")

    mo.hstack(
        [
            mo.as_html(_fig),
            mo.vstack(
                [
                    mo.md(
                        rf"""
    ### The exact matrix at $\theta = {sp.latex(theta_x)}$

    $$R\!\left({sp.latex(theta_x)}\right) \;=\; {sp.latex(R_x)}$$

    Note what the widget did **not** do: above, it printed the literal
    `cos {_deg:.0f}` — a substitution, not a computation. SymPy did the rest, and because
    ${_deg:.0f}^\circ$ arrived as the *exact* rational multiple ${sp.latex(theta_x)}$ of
    $\pi$, the entries came out as ${sp.latex(sp.cos(theta_x))}$ and
    ${sp.latex(sp.sin(theta_x))}$ — closed forms, never a float.

    {check(r"$R^\top R = I$ still", sp.simplify(R_x.T * R_x - I2) == sp.zeros(2))}
    {check(r"$\det R = 1$ still", sp.simplify(R_x.det()) == 1)}

    The grey square is the unit square, the blue one its image: rotated, same area.
    """
                    ),
                ],
                gap=0.4,
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Composition: $R(\alpha)R(\beta) = R(\alpha+\beta)$

    Proven symbolically above. Here it is as a thing you can push around — drag $\alpha$
    and $\beta$ independently and watch the right-hand side stay glued to the left.

    Watch the right-hand side carefully as you drag: it shows `30 + 45`, *unevaluated*.
    That is `TangleLatex` being honest about what it is — a substituting renderer. Adding
    the two is Python's job, one cell down.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    comp = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"R(\tangle{alpha}^\circ)\,R(\tangle{beta}^\circ)"
                r"\;=\;R\big(\tangle{alpha}^\circ + \tangle{beta}^\circ\big)"
            ),
            parameters={
                "alpha": {
                    "value": 30,
                    "min_value": -180,
                    "max_value": 180,
                    "step": 15,
                    "digits": 0,
                    "display": "symbol",
                    "symbol": r"\alpha",
                    "label": "first rotation (degrees)",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
                "beta": {
                    "value": 45,
                    "min_value": -180,
                    "max_value": 180,
                    "step": 15,
                    "digits": 0,
                    "display": "symbol",
                    "symbol": r"\beta",
                    "label": "second rotation (degrees)",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
            },
            reveal_all_on_drag=True,
            editor="inline",
            theme=theme.value,
        )
    )
    comp
    return (comp,)


@app.cell(hide_code=True)
def _(
    GREEN,
    GREY,
    ORANGE,
    Ra,
    Rab,
    Rb,
    al,
    arrow,
    be,
    check,
    comp,
    exact_deg,
    mo,
    np,
    plane,
    sp,
):
    _a = exact_deg(comp.values["alpha"])
    _b = exact_deg(comp.values["beta"])
    _subs = {al: _a, be: _b}
    Ra_x = sp.simplify(Ra.subs(_subs))
    Rb_x = sp.simplify(Rb.subs(_subs))
    Rab_x = sp.simplify(Rab.subs(_subs))
    prod_x = sp.simplify(Ra_x * Rb_x)

    _fig, _ax = plane(extent=1.6, size=(4.0, 4.0))
    _t = np.linspace(0, float(_a + _b), 200)
    _ax.plot(0.85 * np.cos(_t), 0.85 * np.sin(_t), color=GREY, lw=1.0, ls=":")
    arrow(_ax, [1, 0], color=GREY, alpha=0.45, label=r"$e_1$")
    arrow(
        _ax,
        sp.matrix2numpy(Ra_x, dtype=float) @ [1, 0],
        color=ORANGE,
        label=r"$R_\alpha e_1$",
    )
    arrow(
        _ax,
        sp.matrix2numpy(prod_x, dtype=float) @ [1, 0],
        color=GREEN,
        label=r"$R_\beta R_\alpha e_1$",
    )

    mo.hstack(
        [
            mo.as_html(_fig),
            mo.md(
                rf"""
    $$R({sp.latex(_a)})\,R({sp.latex(_b)}) \;=\; {sp.latex(prod_x)}
      \;=\; R\big({sp.latex(sp.simplify(_a + _b))}\big)$$

    {
                    check(
                        rf"the product equals $R({sp.latex(sp.simplify(_a + _b))})$ exactly",
                        sp.simplify(prod_x - Rab_x) == sp.zeros(2),
                    )
                }
    {
                    check(
                        r"and it commutes, $R_\alpha R_\beta = R_\beta R_\alpha$ (rotations of a plane "
                        r"share an axis)",
                        sp.simplify(Ra_x * Rb_x - Rb_x * Ra_x) == sp.zeros(2),
                    )
                }

    Two turns, applied in sequence, land where one turn of the summed angle lands —
    which *is* the pair of angle-sum identities, with the bookkeeping done by matrix
    multiplication instead of by trigonometry.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.5,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Projection, reflection, shear, scaling

    In addition to rotations and scaling operations, the linear operators consist of projection, reflection, and shear transformations. And each of these operators carries a defining property similar to the properties we found for the rotations.

    _The projection_ of a point onto a line defined by a vector $a$ is given by $P_a : V \rightarrow V$. The projection's defining property is that it is idempotent, which means that repeated application of the operator leaves the value unchanged after the first application. In mathematical notation, that is, $P_a^2 = P_a$.

    _The reflection_, $F_a$, of a point across the line defined by $a$ is its own inverse $F_a F_a = F_a^2 = I$, where $I$ is the identity operation. A fancy name for an mapping that has this property is to call it an **involution**.

    _The shear operation_, $H_{k,d}$, slides points parallel to a direction $d$ by an amount proportional to their distance from the line through $d$ — the component orthogonal to $d$, scaled by $k$, is added along $d$: $H_{k,d}\,x = x + k\,(n \cdot x)\,d$ with $n \perp d$. It fixes that line pointwise and preserves area, $\det H = 1$. Its defining property is that $H - I$ squares to zero, $(H_{k,d} - I)^2 = 0$ — a matrix that is the identity plus something nilpotent. A fancy name for this is **unipotent**, and such a shear is called a *transvection*. In dimensions above two the single subscript $d$ no longer suffices: the fixed set is a hyperplane, and $d$ only picks a direction inside it.
    """)
    return


@app.cell(hide_code=True)
def _(I2, J, Matrix, check, mo, simplify, sp, symbols):
    a1, a2, k = symbols("a1 a2 k", real=True)
    a_vec = Matrix([a1, a2])

    P_a = (a_vec * a_vec.T) / (a_vec.T * a_vec)[0]  # projection onto the line through a
    F_a = 2 * P_a - I2  # reflection across that line

    d_vec = Matrix([1, 0])
    n_vec = J * d_vec  # unit normal to the shear line (the x-axis)
    H_k = I2 + k * (d_vec * n_vec.T)  # shear of strength k along d

    mo.md(
        rf"""
    $$P_a = \frac{{a\,a^{{\top}}}}{{a^{{\top}}a}} = {sp.latex(sp.simplify(P_a))}
      \qquad H_k = I + k\,d\,n^{{\top}} = {sp.latex(H_k)}$$

    {check(r"$P_a$ is idempotent, $P_a^2 = P_a$ — projecting twice is projecting once", simplify(P_a * P_a - P_a) == sp.zeros(2))}
    {check(r"$F_a = 2P_a - I$ is an involution, $F_a^2 = I$", simplify(F_a * F_a) == I2)}
    {check(r"$H_k$ is unipotent, $(H_k - I)^2 = 0$ — the shear's defining property", simplify((H_k - I2) * (H_k - I2)) == sp.zeros(2))}
    {check(r"the shear fixes its line, $H_k d = d$", simplify(H_k * d_vec) == d_vec)}
    {check(rf"the shear preserves area, $\det H_k = {sp.latex(sp.simplify(H_k.det()))}$", simplify(H_k.det()) == 1)}
    """
    )
    return H_k, k


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    shear = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"H \;=\; \begin{bmatrix} 1 & \tangle{k} \\[2pt] 0 & 1 \end{bmatrix}"
                r"\qquad \det H = 1"
            ),
            parameters={
                "k": {
                    "value": 1,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,  # quarters -> Rational("0.25") is exact
                    "digits": 2,
                    "label": "shear strength",
                    "color": {"light": "#6f5cbd", "dark": "#b9a8ff"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    shear
    return (shear,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREY,
    H_k,
    ORANGE,
    arrow,
    exact,
    k,
    mo,
    np,
    plane,
    polygon,
    shear,
    sp,
):
    k_x = exact(shear.values["k"])
    H_x = H_k.subs(k, k_x)
    _num = sp.matrix2numpy(H_x, dtype=float)
    _square = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)

    _fig, _ax = plane(extent=3)
    polygon(_ax, _square.T, GREY, lw=1.2, fill=0.06)
    polygon(_ax, (_num @ _square).T, BLUE, lw=2.0)
    arrow(_ax, [1, 0], color=ORANGE, label=r"$d$ (fixed)")
    arrow(_ax, _num @ [0, 1], color=BLUE, label=r"$He_2$")

    mo.hstack(
        [
            mo.as_html(_fig),
            mo.md(
                rf"""
    $$H = {sp.latex(H_x)} \qquad \det H = {sp.latex(sp.simplify(H_x.det()))}$$

    Drag $k$ and watch the picture: the square slides into a parallelogram, the orange
    vector $d$ never moves, and the shaded area never changes. That is what
    $\det H = 1$ *means*, and the exact determinant above never drifts off $1$ no matter
    how far you drag — the invariance is algebraic, not numerical.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Dot and wedge: the Pythagorean identity, proven (notebook 3)

    The dot product ($\langle u,v\rangle = \lvert u\rvert\lvert v\rvert\cos\theta$,
    symmetric) and the wedge ($u\wedge v = \lvert u\rvert\lvert v\rvert\sin\theta\,e_1\wedge e_2$,
    a bivector representing signed area, antisymmetric) are the two halves of multiplying two
    arrows. The squared dot product plus the squared magnitude of the wedge add to the product
    of the squared lengths — which is $\cos^2+\sin^2=1$ in disguise.
    """)
    return


@app.cell(hide_code=True)
def _(Matrix, check, mo, simplify, sp, symbols):
    u1, u2, v1, v2 = symbols("u1 u2 v1 v2", real=True)
    u_sym, v_sym = Matrix([u1, u2]), Matrix([v1, v2])

    dot_sym = (u_sym.T * v_sym)[0]
    wedge_sym = u1 * v2 - u2 * v1  # signed area = the 2x2 determinant [u v]
    _lhs = dot_sym**2 + wedge_sym**2
    _rhs = (u_sym.T * u_sym)[0] * (v_sym.T * v_sym)[0]

    # NOTE: the Jupyter original used (1,2,2) and (2,0,-1) here, but those are
    # *orthogonal* -- their dot product is 2 + 0 - 2 = 0, so the "exact angle" it
    # advertises collapses to a dull pi/2 (and its summary table's claim of arccos(5/9)
    # matches neither). (2,2,1) has the same norm 3 and gives a genuinely irrational
    # angle, arccos(8/9), which is the point the example was reaching for.
    _uu, _vv = Matrix([1, 2, 2]), Matrix([2, 2, 1])
    _cos_ang = (_uu.T * _vv)[0] / (_uu.norm() * _vv.norm())

    mo.md(
        rf"""
    $$\langle u,v\rangle^2 + \lvert u\wedge v\rvert^2 - \lVert u\rVert^2\lVert v\rVert^2
      \;=\; {sp.latex(simplify(_lhs - _rhs))}$$

    {check(r"identically zero, for all $u,v$ — the Pythagorean identity", simplify(_lhs - _rhs) == 0)}
    {check(r"$u \wedge v = \det[\,u\;\;v\,] e_1 \wedge e_2$  — the wedge generates the determinant", simplify(wedge_sym - u_sym.row_join(v_sym).det()) == 0)}

    And an exact angle where `numpy` would only ever hand back a decimal — between
    $(1,2,2)$ and $(2,2,1)$ (both of norm $3$):

    $$\cos\vartheta = {sp.latex(sp.nsimplify(_cos_ang))}
      \qquad \vartheta = {sp.latex(sp.acos(_cos_ang))} \approx {float(sp.acos(_cos_ang)):.4f}\ \text{{rad}}$$
    """
    )
    return dot_sym, u1, u2, u_sym, v1, v2, v_sym, wedge_sym


@app.cell
def _(u_sym, v_sym):
    u_sym, v_sym, u_sym.T * v_sym
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    uv = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"u = \begin{bmatrix}\tangle{u1}\\[2pt]\tangle{u2}\end{bmatrix}"
                r"\qquad v = \begin{bmatrix}\tangle{v1}\\[2pt]\tangle{v2}\end{bmatrix}"
            ),
            parameters={
                "u1": {
                    "value": 2,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,
                    "digits": 2,
                    "label": "u_x",
                    "color": {"light": "#b91c1c", "dark": "#f87171"},
                },
                "u2": {
                    "value": 0.5,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,
                    "digits": 2,
                    "label": "u_y",
                    "color": {"light": "#b91c1c", "dark": "#f87171"},
                },
                "v1": {
                    "value": 0.75,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,
                    "digits": 2,
                    "label": "v_x",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
                "v2": {
                    "value": 1.75,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,
                    "digits": 2,
                    "label": "v_y",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    uv
    return (uv,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREEN,
    RED,
    arrow,
    check,
    dot_sym,
    exact,
    mo,
    plane,
    polygon,
    sp,
    u1,
    u2,
    uv,
    v1,
    v2,
    wedge_sym,
):
    _s = {
        u1: exact(uv.values["u1"]),
        u2: exact(uv.values["u2"]),
        v1: exact(uv.values["v1"]),
        v2: exact(uv.values["v2"]),
    }
    dot_x = sp.simplify(dot_sym.subs(_s))
    wedge_x = sp.simplify(wedge_sym.subs(_s))
    _un = [float(_s[u1]), float(_s[u2])]
    _vn = [float(_s[v1]), float(_s[v2])]

    _fig, _ax = plane(extent=3)
    polygon(
        _ax,
        [[0, 0], _un, [_un[0] + _vn[0], _un[1] + _vn[1]], _vn],
        BLUE,
        lw=2.0,
        fill=0.15,
    )
    arrow(_ax, _un, color=RED, label=r"$u$")
    arrow(_ax, _vn, color=GREEN, label=r"$v$")

    _nu2 = sp.simplify(_s[u1] ** 2 + _s[u2] ** 2)
    _nv2 = sp.simplify(_s[v1] ** 2 + _s[v2] ** 2)

    mo.hstack(
        [
            mo.as_html(_fig),
            mo.md(
                rf"""
    $$\langle u,v\rangle = {sp.latex(dot_x)} \qquad u\wedge v = {sp.latex(wedge_x)}e_1\wedge e_2$$

    $$\underbrace{{{sp.latex(dot_x**2)}}}_{{\langle u,v\rangle^2}} +
      \underbrace{{{sp.latex(wedge_x**2)}}}_{{\lvert u\wedge v\rvert^2}} =
      {sp.latex(sp.simplify(dot_x**2 + wedge_x**2))} =
      \underbrace{{{sp.latex(_nu2)} \cdot {sp.latex(_nv2)}}}_{{\lVert u\rVert^2\lVert v\rVert^2}}$$

    {check("the identity holds at these exact values", sp.simplify(dot_x**2 + wedge_x**2 - _nu2 * _nv2) == 0)}

    The shaded parallelogram's signed area is represented by $u\wedge v$. Drag
    $v$ until it lines up with $u$: the area collapses to $0$ while the dot
    product swells to $\lVert u\rVert\lVert v\rVert$. The two halves trade off
    along a circle — because they are $\cos$ and $\sin$ wearing different
    clothes. Cross $v$ over to the other side of $u$ and the wedge goes
    **negative**; the dot product does not notice. That sign is the orientation
    the determinant tracks and the length cannot see.  """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Determinant and trace (notebook 4), exactly

    The determinant is **multiplicative** and equals the factor by which signed area
    scales. The trace is the *first-order* area rate,
    $\operatorname{tr}T = \frac{d}{dt}\big|_0 \det(I + tT)$ — the determinant's derivative
    at the identity. Both hold for symbolic matrices, and the section closes with
    $\det e^{tT} = e^{t\operatorname{tr}T}$.
    """)
    return


@app.cell(hide_code=True)
def _(Matrix, check, eye, mo, simplify, sp, symbols):
    ma, mb, mc, md, me, mf, mg, mh = symbols("a b c d e f g h", real=True)
    t = symbols("t", real=True)

    A_sym = Matrix([[ma, mb], [mc, md]])
    B_sym = Matrix([[me, mf], [mg, mh]])
    area_t = (eye(2) + t * A_sym).det()  # signed area of the image of the unit square

    T_tri = Matrix(
        [[1, 2], [0, 3]]
    )  # concrete & triangular, so the matrix exp is clean
    exp_T = (t * T_tri).exp()

    kk = symbols("kappa", real=True)
    B_kk = Matrix([[1, kk], [kk, 4]])

    mo.md(
        rf"""
    {check(r"$\det(AB) = \det A \cdot \det B$, for arbitrary $2\times2$ symbolic $A,B$", simplify((A_sym * B_sym).det() - A_sym.det() * B_sym.det()) == 0)}

    $$\det(I + tT) = {sp.latex(sp.expand(area_t))}
      \qquad \frac{{d}}{{dt}}\Big|_{{0}} \det(I + tT) = {sp.latex(sp.diff(area_t, t).subs(t, 0))}
      = \operatorname{{tr}} T$$

    The trace is the **linear** part of the area change: at first order, area grows by
    $a + d$ and the off-diagonal $bc$ term only shows up at order $t^2$.

    $$e^{{tT}} = {sp.latex(exp_T)}
      \qquad \det e^{{tT}} = {sp.latex(simplify(exp_T.det()))}
      \qquad e^{{t\operatorname{{tr}}T}} = {sp.latex(sp.exp(t * T_tri.trace()))}$$

    {check(r"$\det e^{tT} = e^{t\operatorname{tr}T}$ — the trace exponentiates into the determinant", simplify(exp_T.det() - sp.exp(t * T_tri.trace())) == 0)}

    And a parameter living *inside* the matrix, with SymPy solving for where it degenerates:

    $$\det{sp.latex(B_kk)} = {sp.latex(B_kk.det())}
      \;\longrightarrow\; \text{{singular at }} \kappa \in {sp.latex(sp.solve(B_kk.det(), kk))}$$
    """
    )
    return (A_sym,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Change of basis, and discovering $\mathbb{C}$ (notebook 5)

    Writing an operator in a different basis is the **similarity** $A \mapsto P^{-1}AP$.
    The determinant and trace are basis-independent — they belong to the operator, not to
    its matrix.
    """)
    return


@app.cell(hide_code=True)
def _(A_sym, Matrix, check, mo, simplify, sp):
    P_cob = Matrix([[2, 1], [1, 1]])  # any invertible change of basis
    A_similar = sp.simplify(P_cob.inv() * A_sym * P_cob)

    mo.md(
        rf"""
    $$P^{{-1}}AP = {sp.latex(A_similar)}$$

    {check(r"$\det$ preserved under similarity", simplify(A_similar.det() - A_sym.det()) == 0)}
    {check(r"$\operatorname{tr}$ preserved under similarity", simplify(A_similar.trace() - A_sym.trace()) == 0)}

    Every individual entry changed; the two invariants did not.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The capstone: $xI + yJ$ **is** $\mathbb{C}$

    The "scale-and-rotate" operators — a real multiple of the identity plus a real multiple
    of the quarter-turn — are closed and commutative under multiplication, with
    $\det = x^2+y^2$. Send $xI + yJ \leftrightarrow x + iy$ and matrix multiplication
    *becomes* complex multiplication, exactly.
    """)
    return


@app.cell
def _(I2, J, check, mo, simplify, sp, symbols, trigsimp):
    zx, zy, zx2, zy2, phi = symbols("x y x_2 y_2 phi", real=True)

    def Z(xx, yy):
        return xx * I2 + yy * J

    Z_prod = Z(zx, zy) * Z(zx2, zy2)
    exp_phiJ = trigsimp((phi * J).exp())

    mo.md(
        rf"""
    $$(xI + yJ)(x_2 I + y_2 J) = {sp.latex(Z_prod)}$$

    {
            check(
                r"equals $Z(xx_2 - yy_2,\; xy_2 + x_2y)$ — *exactly* the rule "
                r"$(x+iy)(x_2+iy_2) = (xx_2 - yy_2) + (xy_2 + x_2y)\,i$",
                simplify(Z_prod - Z(zx * zx2 - zy * zy2, zx * zy2 + zx2 * zy))
                == sp.zeros(2),
            )
        }
    {
            check(
                r"commutative, $Z_1Z_2 = Z_2Z_1$ (unusual for matrices!)",
                simplify(Z_prod - Z(zx2, zy2) * Z(zx, zy)) == sp.zeros(2),
            )
        }
    {
            check(
                rf"the modulus squared is the determinant, $\det Z = {sp.latex(sp.simplify(Z(zx, zy).det()))}$",
                simplify(Z(zx, zy).det() - (zx**2 + zy**2)) == 0,
            )
        }

    $$e^{{\phi J}} = {sp.latex(exp_phiJ)}$$

    {
            check(
                r"$e^{\phi J} = \cos\phi\,I + \sin\phi\,J = R(\phi)$ — Euler's formula $e^{i\phi} = \cos\phi + i\sin\phi$, as a matrix identity",
                trigsimp((phi * J).exp() - (sp.cos(phi) * I2 + sp.sin(phi) * J))
                == sp.zeros(2),
            )
        }
    """
    )
    return (Z,)


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    znum = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"Z \;=\; \tangle{x}\,I + \tangle{y}\,J"
                r"\;=\; \begin{bmatrix}\tangle{x} & -\tangle{y}\\[2pt]"
                r"\tangle{y} & \tangle{x}\end{bmatrix}"
                r"\;\longleftrightarrow\; \tangle{x} + \tangle{y}\,i"
            ),
            parameters={
                "x": {
                    "value": 1.5,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,
                    "digits": 2,
                    "label": "real part",
                    "color": {"light": "#246bce", "dark": "#75a7ff"},
                },
                "y": {
                    "value": 1,
                    "min_value": -3,
                    "max_value": 3,
                    "step": 0.25,
                    "digits": 2,
                    "label": "imaginary part",
                    "color": {"light": "#a23b78", "dark": "#f08bc2"},
                },
            },
            editor="inline",
            theme=theme.value,
        )
    )
    znum
    return (znum,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREEN,
    GREY,
    PURPLE,
    Z,
    arrow,
    check,
    exact,
    mo,
    np,
    plane,
    sp,
    znum,
):
    x_x, y_x = exact(znum.values["x"]), exact(znum.values["y"])
    Z_x = Z(x_x, y_x)
    _mod = sp.sqrt(sp.simplify(Z_x.det()))
    _arg = sp.atan2(y_x, x_x)
    _num = sp.matrix2numpy(Z_x, dtype=float)

    _fig, _ax = plane(extent=3)
    _t = np.linspace(0, float(_arg), 200)
    _r = float(_mod)
    _ax.plot(_r * np.cos(_t), _r * np.sin(_t), color=GREY, lw=1.0, ls=":")
    arrow(_ax, [1, 0], color=GREY, alpha=0.5, label=r"$1$")
    arrow(_ax, [0, 1], color=GREY, alpha=0.5, label=r"$i$")
    # Z e_1 = (x, y) = z exactly -- the first column of the matrix *is* the complex
    # number, so one arrow serves as both. Drawing them separately would just stack
    # two arrowheads in the same place.
    arrow(_ax, [float(x_x), float(y_x)], color=PURPLE, label=r"$z = Z e_1$", lw=2.6)
    arrow(_ax, _num @ [0, 1], color=GREEN, label=r"$Z e_2$", alpha=0.85)
    _ax.plot([0, float(x_x)], [float(y_x)] * 2, color=BLUE, lw=0.9, ls="--", alpha=0.6)
    _ax.plot([float(x_x)] * 2, [0, float(y_x)], color=BLUE, lw=0.9, ls="--", alpha=0.6)

    mo.hstack(
        [
            mo.as_html(_fig),
            mo.md(
                rf"""
    $$Z = {sp.latex(Z_x)} \qquad \det Z = {sp.latex(sp.simplify(Z_x.det()))}
      \qquad \lvert z\rvert = {sp.latex(sp.simplify(_mod))}$$

    $$\arg z = {sp.latex(sp.simplify(_arg))} \approx {np.degrees(float(_arg)):.1f}^\circ$$

    {check(rf"$\det Z = x^2+y^2 = \lvert z\rvert^2$", sp.simplify(Z_x.det() - (x_x**2 + y_x**2)) == 0)}
    {check(r"$Z$ acts as *scale by $\lvert z\rvert$, then rotate by $\arg z$*", sp.simplify(Z_x - _mod * (sp.cos(_arg) * sp.eye(2) + sp.sin(_arg) * sp.Matrix([[0, -1], [1, 0]]))) == sp.zeros(2))}

    Drag $x$ and $y$, and notice that the purple arrow carries **two labels at once**.
    $Ze_1 = (x, y) = z$ identically: the *first column of the matrix is the complex
    number*. So the arrow showing where $Z$ sends $e_1$ and the arrow showing $z$ as a
    point of the plane are the same arrow — that coincidence is the whole isomorphism
    $xI + yJ \leftrightarrow x + iy$, drawn.

    The green $Ze_2$ then has no freedom left: it is forced to be $z$ turned a quarter
    turn, always perpendicular and always the same length. Two real numbers in, two
    arrows out, one of them determined. Set $x=0,\,y=1$ to recover $J$ itself: a pure
    quarter-turn, $\lvert z\rvert = 1$, $\det = 1$, $i^2 = -1$.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Every headline result of the series, re-proven exactly — and now each one has a handle
    on it.

    | result | nb | how SymPy shows it | what you drag |
    |---|---|---|---|
    | $J^2=-I$; $R(\theta)$ orthogonal; $R(\alpha)R(\beta)=R(\alpha{+}\beta)$ | 2 | `trigsimp(...) == 0` | $\theta$; $\alpha,\beta$ |
    | $P_a$ idempotent, $F_a$ an involution, shear preserves area | 2 | symbolic direction $a$ | shear strength $k$ |
    | $\langle u,v\rangle^2 + \lvert u\wedge v\rvert^2 = \lVert u\rVert^2\lVert v\rVert^2$; wedge $\approx \det[u\,v]$ | 3 | `simplify(lhs - rhs) == 0` | $u,v$ components |
    | $\det(AB)=\det A\det B$; $\operatorname{tr}T = \frac{d}{dt}\rvert_0\det(I{+}tT)$; $\det e^{tT}=e^{t\operatorname{tr}T}$ | 4 | symbolic $A,B,T$; `diff`, `.exp()` | — |
    | $\det,\operatorname{tr}$ similarity-invariant; $xI+yJ \cong \mathbb{C}$; $e^{\phi J}=R(\phi)$ | 5 | `P.inv()*A*P`, exact complex product | $x, y$ |

    ### Two lessons, one from each layer

    **From SymPy:** a `simplify(...) == 0` on a matrix of symbols is a proof *for all
    inputs*, where `np.allclose` only ever checks one sample. Exact arithmetic also keeps
    the geometry legible — angles stay $\arccos\frac{8}{9}$, determinants stay $x^2+y^2$ —
    instead of dissolving into floating-point noise.

    **From the tangled formulas:** the reason this works at all is that the dragging and
    the proving are cleanly separated. `TangleLatex` never computes; it hands marimo a
    number, marimo re-runs the dependent cells, and SymPy does every bit of the math. And
    because the step sizes were chosen to land on exact rationals, nothing is lost in
    translation from mouse to surd. Use them together: SymPy to *prove and understand*,
    the widget to *feel which knob does what*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Notes on the translation from Jupyter

    Three things had to change, and each is a real difference between the two formats
    rather than a cosmetic one:

    1. **`print(...)` became returned values.** A marimo cell renders its *last
       expression*. The original's dozens of `print("claim:", bool)` lines are now a
       `check()` helper feeding into a single `mo.md(...)`, so the verdicts render as real
       math instead of monospace text. This is the same job
       `clmmathtools.viz.show_expr` does in the Jupyter version — marimo's `mo.md` plus
       `sp.latex` covers it natively, so the notebook needs no `clmmathtools` import.
    2. **Every global is defined exactly once.** Marimo builds a DAG from the variable
       names, so it forbids the notebook habit of rebinding `A`, `P`, and `a` in cell after
       cell. Hence `A_sym` / `A_similar` / `P_a` / `P_cob` / `a_vec`, and a leading
       underscore (`_fig`, `_ax`, `_num`) on anything cell-local. Restrictive at first,
       then quietly clarifying: nothing depends on execution order any more, so there is no
       such thing as a stale cell.
    3. **vedo gave way to matplotlib.** `Plane2D.display()` opens a blocking VTK window,
       which cannot work in a cell that re-runs on every mouse drag. A returned matplotlib
       `Figure` re-renders instantly, so the reactive loop stays fluid. The named color
       constants carried over unchanged. If you want the vedo look back, the original's
       `ipydisplay()` path — offscreen `Plotter` → `screenshot(asarray=True)` → PIL — is
       the piece to reuse; return the PIL image as the cell's value.

    Also worth knowing: **marimo notebooks *are* `.py` files**, so no jupytext pairing is
    needed here. This file is the notebook — plain Python, diffable, no base64 blobs, no
    git-LFS. Export a static copy with:

    ```
    uv run marimo export html-wasm --sandbox --mode edit \
        GeometricLinearAlgebra/00_SymPy_Symbolic_LinearAlgebra_marimo.py -o site/
    ```

    `html-wasm` is the one to reach for: it ships the notebook as Pyodide, so the tangled
    formulas keep working in a plain static browser page with no server. A regular
    `marimo export html` freezes the current outputs instead.
    """)
    return


if __name__ == "__main__":
    app.run()
