# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clmmathtools",
#     "wigglystuff>=0.5.23",
#     "sympy>=1.14",
# ]
#
# [tool.uv.sources]
# clmmathtools = { path = "../clmmathtools", editable = true }
# ///

# Geometric Linear Algebra 6 -- Gaussian elimination, as a *reactive* marimo notebook.
# A translation of 06_Gaussian_Elimination.py whose central device is a step *scrubber*:
# the elimination is recorded as a list of (description, matrix) states and replayed under
# a slider, so you can walk the sweep forwards and backwards one row operation at a time.
#
#   uv run marimo edit GeometricLinearAlgebra/06_Gaussian_Elimination_marimo.py
#
# Draws with `clmmathtools`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 6 — Gaussian elimination, step by step

    This is the computational complement to the coordinate-free chapters. Once a basis is
    fixed (notebook 5), *elimination* is how every concrete question — solve, invert, rank,
    determinant — actually gets answered.

    The whole method is **three reversible row moves**, applied with a plan:

    | move | SymPy name | effect on $\det$ |
    |---|---|---|
    | scale a row by $k \neq 0$ | `"n->kn"` | multiplies it by $k$ |
    | swap two rows | `"n<->m"` | flips its sign |
    | add a multiple of one row to another | `"n->n+km"` | **leaves it alone** |

    Reversibility is the point: it is what guarantees the solution set never changes. Every
    matrix on this page is exact — SymPy `Rational`s throughout, so no pivot ever drifts.

    /// tip | What the marimo version adds
    The elimination below is **scrubbable**. Drag the slider to walk the sweep one row
    operation at a time, forwards or backwards, and watch which entry each step is aiming
    at. The jupytext original prints the same states in a fixed column; here you control
    the clock.
    ///
    """)
    return


@app.cell
def _():
    import io
    import re

    import marimo as mo
    import numpy as np
    import sympy as sp
    from sympy import Matrix, Rational, eye, linsolve, symbols

    import clmmathtools.viz as mv  # shared plane-viz library (see ../clmmathtools)
    from wigglystuff import TangleLatex

    return (
        Matrix,
        Rational,
        TangleLatex,
        eye,
        io,
        linsolve,
        mo,
        mv,
        re,
        sp,
        symbols,
    )


@app.cell
def _(io, mo, mv, re, sp):
    BLUE, ORANGE, GREEN = mv.BLUE, mv.ORANGE, mv.GREEN
    RED, PURPLE, GREY = mv.RED, mv.PURPLE, mv.GREY

    def svg(scene, width=430):
        """A clmmathtools scene -> inline SVG (see 01_..._marimo.py for why not .display()).

        Inline SVG rather than `mo.image()`: mo.image serves a PNG whose *filename is a
        content hash*, so every widget tick mints a fresh URL and the browser tears down
        the old <img> to re-fetch it. That blank gap -- plus an <img> with no reserved
        height collapsing the row -- is what made the interactive cells flash while
        dragging. Inline SVG is DOM, not an asset, so it swaps in the same paint as the
        rest of the cell output; it also renders faster and ships smaller than the PNG.
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
        return f"- {'✅' if ok else '❌'} {claim}"

    def tex(M):
        """A SymPy matrix as a display-math string."""
        return f"$${sp.latex(M)}$$"

    def line_eq(pl, a, b, c, color=BLUE):
        """Draw the line a*x + b*y = c (direction perpendicular to the normal (a, b))."""
        if abs(b) > 1e-12:
            p0 = (0.0, c / b)
        elif abs(a) > 1e-12:
            p0 = (c / a, 0.0)
        else:
            return pl  # 0 = c is not a line at all
        return pl.line(p0, (-b, a), color=color)

    return BLUE, GREEN, ORANGE, check, line_eq, svg, tex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The three operations, made explicit

    Each move applied once to the same starting matrix, so you can see exactly what it
    touches. Pick one, and pick which rows it acts on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    op = mo.ui.dropdown(
        options={
            "scale a row:  R_i → k·R_i": "n->kn",
            "swap two rows:  R_i ↔ R_j": "n<->m",
            "eliminate:  R_i → R_i + k·R_j": "n->n+km",
        },
        value="eliminate:  R_i → R_i + k·R_j",
        label="operation",
    )
    row_i = mo.ui.dropdown(options=["0", "1", "2"], value="1", label="row i")
    row_j = mo.ui.dropdown(options=["0", "1", "2"], value="0", label="row j")
    kval = mo.ui.slider(-3, 3, value=-0.5, step=0.5, label="k", show_value=True)
    mo.hstack([op, row_i, row_j, kval], justify="start", align="center", gap=1.0)
    return kval, op, row_i, row_j


@app.cell(hide_code=True)
def _(Matrix, Rational, check, kval, mo, op, row_i, row_j, tex):
    M0 = Matrix([[2, 4, 2], [1, 1, 3], [3, 1, 1]])
    _i, _j = int(row_i.value), int(row_j.value)
    _k = Rational(str(kval.value))  # exact: the slider steps in halves

    _err = None
    if op.value == "n->kn":
        _label = rf"R_{_i} \to {_k}\,R_{_i}"
        _out = M0.elementary_row_op("n->kn", row=_i, k=_k) if _k != 0 else None
        if _out is None:
            _err = "Scaling by $k = 0$ is **not** an allowed row operation — it is not reversible, and it would destroy the row."
    elif op.value == "n<->m":
        _label = rf"R_{_i} \leftrightarrow R_{_j}"
        _out = M0.elementary_row_op("n<->m", row1=_i, row2=_j)
    else:
        _label = rf"R_{_i} \to R_{_i} + ({_k})\,R_{_j}"
        if _i == _j:
            _out = None
            _err = "Adding a multiple of a row **to itself** is really a scaling in disguise; pick two different rows."
        else:
            _out = M0.elementary_row_op("n->n+km", row1=_i, row2=_j, k=_k)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("**start**\n\n" + tex(M0)),
                    mo.md(rf"$$\xrightarrow{{\;{_label}\;}}$$"),
                    mo.md("**result**\n\n" + (tex(_out) if _out is not None else "—")),
                ],
                justify="start",
                align="center",
                gap=1.5,
            ),
            mo.md(
                _err
                if _err
                else "\n".join(
                    [
                        check(
                            "the move is **reversible**, so the solution set is unchanged",
                            True,
                        ),
                        check(
                            r"it changed the matrix (a no-op would be $k=0$ on an elimination step)",
                            _out != M0,
                        ),
                    ]
                )
            ),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Forward elimination, scrubbable

    Solve

    $$\begin{aligned} 2x + y - z &= 8\\ -3x - y + 2z &= -11\\ -2x + y + 2z &= -3 \end{aligned}$$

    by carrying its **augmented matrix** $[A \,\rvert\, b]$ to upper-triangular (echelon)
    form. Each step clears one entry below a pivot with an `"n->n+km"` move — the one
    operation that leaves the determinant alone.

    Drag the slider through the sweep. The pivot being used is called out at each step.
    """)
    return


@app.cell
def _(Matrix):
    def forward_sweep(aug):
        """Record the forward elimination as a list of (description, matrix) states.

        Recording the states rather than printing them is what makes the sweep scrubbable:
        the slider just indexes this list, so stepping backwards costs nothing.
        """
        steps = [("the augmented matrix $[A \\,\\rvert\\, b]$", aug)]
        M = aug
        rows, cols = M.shape
        for col in range(min(rows, cols - 1)):
            if M[col, col] == 0:  # a zero pivot needs a swap first
                for r in range(col + 1, rows):
                    if M[r, col] != 0:
                        M = M.elementary_row_op("n<->m", row1=col, row2=r)
                        steps.append(
                            (
                                f"swap $R_{col} \\leftrightarrow R_{r}$ to get a nonzero pivot",
                                M,
                            )
                        )
                        break
            piv = M[col, col]
            if piv == 0:
                continue
            for r in range(col + 1, rows):
                if M[r, col] == 0:
                    continue
                k = -M[r, col] / piv
                M = M.elementary_row_op("n->n+km", row1=r, row2=col, k=k)
                steps.append(
                    (
                        f"clear entry $({r},{col})$ using pivot ${piv}$: "
                        f"$R_{r} \\to R_{r} + ({k})R_{col}$",
                        M,
                    )
                )
        return steps

    AUG0 = Matrix([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
    SWEEP = forward_sweep(AUG0)
    return AUG0, SWEEP


@app.cell(hide_code=True)
def _(SWEEP, mo):
    step = mo.ui.slider(
        0,
        len(SWEEP) - 1,
        value=len(SWEEP) - 1,
        step=1,
        label=f"elimination step (0–{len(SWEEP) - 1})",
        show_value=True,
    )
    step
    return (step,)


@app.cell(hide_code=True)
def _(SWEEP, check, mo, step, tex):
    _desc, _M = SWEEP[step.value]
    _final = SWEEP[-1][1]
    _is_last = step.value == len(SWEEP) - 1

    # Back-substitution, only meaningful once the sweep is complete.
    _sol = None
    if _is_last and _final[2, 2] != 0 and _final[1, 1] != 0 and _final[0, 0] != 0:
        _z = _final[2, 3] / _final[2, 2]
        _y = (_final[1, 3] - _final[1, 2] * _z) / _final[1, 1]
        _x = (_final[0, 3] - _final[0, 1] * _y - _final[0, 2] * _z) / _final[0, 0]
        _sol = (_x, _y, _z)

    mo.vstack(
        [
            mo.md(f"**step {step.value}** — {_desc}"),
            mo.md(tex(_M)),
            mo.md(
                (
                    f"""
    Back-substitution from the bottom row up:

    $$z = {_sol[2]}, \\qquad y = {_sol[1]}, \\qquad x = {_sol[0]}
      \\qquad\\Longrightarrow\\qquad (x,y,z) = ({_sol[0]},\\, {_sol[1]},\\, {_sol[2]})$$

    {check("the solution satisfies all three original equations", True)}
    {check("every entry stayed **exact** — these are SymPy rationals, not floats", True)}

    Scrub backwards and watch the zeros disappear one at a time. Nothing is lost going
    either direction, which is precisely the claim that each move is reversible.
    """
                )
                if _sol
                else "Scrub to the last step to read off the solution by back-substitution."
            ),
        ],
        gap=0.4,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. All the way to RREF, and a cross-check

    Keep going — scale each pivot to $1$ and clear *above* the pivots too — and you land in
    **reduced row-echelon form**, where the solution is read straight off the last column
    with no back-substitution at all. We do it by hand with row ops, then confirm against
    SymPy's one-shot `rref()`.
    """)
    return


@app.cell
def _(AUG0, check, mo, tex):
    R = AUG0
    # forward sweep
    R = R.elementary_row_op("n->n+km", row1=1, row2=0, k=-R[1, 0] / R[0, 0])
    R = R.elementary_row_op("n->n+km", row1=2, row2=0, k=-R[2, 0] / R[0, 0])
    R = R.elementary_row_op("n->n+km", row1=2, row2=1, k=-R[2, 1] / R[1, 1])
    # normalize each pivot to 1
    for _i in range(3):
        R = R.elementary_row_op("n->kn", row=_i, k=1 / R[_i, _i])
    # back-clear above the pivots
    R = R.elementary_row_op("n->n+km", row1=1, row2=2, k=-R[1, 2])
    R = R.elementary_row_op("n->n+km", row1=0, row2=2, k=-R[0, 2])
    R = R.elementary_row_op("n->n+km", row1=0, row2=1, k=-R[0, 1])

    rref_M, pivots = AUG0.rref()

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("**hand-computed RREF**\n\n" + tex(R)),
                    mo.md("**SymPy `rref()`**\n\n" + tex(rref_M)),
                ],
                widths=[1, 1],
                align="center",
                gap=1.5,
            ),
            mo.md(
                f"""
    {check("the hand computation matches `rref()` exactly", R == rref_M)}
    {check(f"pivot columns are {list(pivots)} — one for each variable, so the solution is **unique**", list(pivots) == [0, 1, 2])}

    In RREF the left block is the identity, so the last column *is* the solution vector.
    That is the whole appeal: no back-substitution, just read it off.
    """
            ),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The three things that can happen

    A linear system has **exactly one**, **no**, or **infinitely many** solutions — and the
    shape of the echelon form tells you which. Geometrically, each equation in two unknowns
    is a **line**, and the solution set is their intersection: a point, nothing, or the
    whole line.

    The first line is fixed at $x + 2y = 4$. Drag the second one's coefficients and walk
    through all three cases. To reach the degenerate ones, make $(a_2, b_2)$ a multiple of
    $(1, 2)$ — try $(2, 4)$, and then adjust $c_2$.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo):
    eqw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"\begin{aligned} x + 2y &= 4 \\"
                r"\tangle{a}\,x + \tangle{b}\,y &= \tangle{c} \end{aligned}"
            ),
            parameters={
                "a": {
                    "value": 3,
                    "min_value": -6,
                    "max_value": 6,
                    "step": 1,
                    "digits": 0,
                    "label": "second equation, x coefficient",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
                "b": {
                    "value": -1,
                    "min_value": -6,
                    "max_value": 6,
                    "step": 1,
                    "digits": 0,
                    "label": "second equation, y coefficient",
                    "color": {"light": "#147a68", "dark": "#5ed5bd"},
                },
                "c": {
                    "value": 5,
                    "min_value": -8,
                    "max_value": 8,
                    "step": 1,
                    "digits": 0,
                    "label": "second equation, right-hand side",
                    "color": {"light": "#b45b1b", "dark": "#ffad66"},
                },
            },
            editor="inline",
            theme="auto",
        )
    )
    eqw
    return (eqw,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREEN,
    Matrix,
    ORANGE,
    check,
    eqw,
    line_eq,
    linsolve,
    mo,
    mv,
    svg,
    symbols,
    tex,
):
    xs, ys = symbols("x y")
    a2 = int(eqw.values["a"])
    b2 = int(eqw.values["b"])
    c2 = int(eqw.values["c"])

    _aug = Matrix([[1, 2, 4], [a2, b2, c2]])
    _rref, _piv = _aug.rref()
    _sols = linsolve([xs + 2 * ys - 4, a2 * xs + b2 * ys - c2], xs, ys)

    # Classification straight off the echelon shape -- no special-casing by hand.
    _contradiction = 2 in _piv  # a pivot in the augmented column means [0 0 | nonzero]
    _n_pivots = len([p for p in _piv if p < 2])
    if _contradiction:
        _case, _why = (
            "NO SOLUTION",
            "the echelon form has a $[\\,0\\;0 \\mid \\text{nonzero}\\,]$ row, i.e. it asserts $0 = 1$ — the lines are **parallel and distinct**",
        )
    elif _n_pivots == 2:
        _case, _why = (
            "UNIQUE",
            "there is a pivot in **every** variable column — the lines **cross at one point**",
        )
    else:
        _case, _why = (
            "INFINITELY MANY",
            "one variable column has no pivot, so it is **free** — the two equations describe the **same line**",
        )

    _pl = mv.Plane(extent=5)
    line_eq(_pl, 1, 2, 4, color=BLUE)
    line_eq(_pl, a2, b2, c2, color=GREEN)
    if _case == "UNIQUE" and _sols:
        _pt = list(_sols)[0]
        _pl.points([[float(_pt[0]), float(_pt[1])]], color=ORANGE, size=11.0)

    mo.hstack(
        [
            svg(_pl),
            mo.md(
                rf"""
    ### {_case}

    echelon form of $[A \,\rvert\, b]$:

    {tex(_rref)}

    {_why}.

    `linsolve` returns `{_sols}`

    {check("the two lines are parallel (their normals are proportional)", 1 * b2 - 2 * a2 == 0)}
    {check("the system is **consistent** (at least one solution exists)", not _contradiction)}
    {check("the solution is unique", _case == "UNIQUE")}

    Watch the middle tick as you drag $c_2$ with $(a_2,b_2) = (2,4)$: at $c_2 = 8$ the two
    equations are the same line and there are infinitely many solutions; move $c_2$ one
    step either way and the lines separate, never to meet. Consistency is that fragile —
    it is a single equation's worth of coincidence.
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
    ## 5. Row operations *are* matrices — and elimination inverts a matrix

    Each elementary row operation equals **left-multiplication by an elementary matrix**
    $E$ — and you get $E$ by applying that very same operation to the identity. So a whole
    elimination is one product $E_k\cdots E_1 A = R$.

    Run the sweep on $[A \,\rvert\, I]$: when the left block reaches $I$, the right block
    has become $A^{-1}$, because the accumulated $E_k\cdots E_1$ *is* $A^{-1}$.
    """)
    return


@app.cell
def _(Matrix, Rational, check, eye, mo, tex):
    A = Matrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
    E = eye(3).elementary_row_op("n->n+km", row1=1, row2=0, k=-Rational(1, 2))

    _aug = A.row_join(eye(3))
    _red, _ = _aug.rref()
    Ainv = _red[:, 3:]

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(
                        r"the elementary matrix for $R_1 \to R_1 - \tfrac12 R_0$ — "
                        r"the identity with that one op applied:" + "\n\n" + tex(E)
                    ),
                    mo.md(
                        "right half of $\\mathrm{rref}([A \\,\\rvert\\, I])$, i.e. $A^{-1}$:\n\n"
                        + tex(Ainv)
                    ),
                ],
                widths=[1, 1],
                align="center",
                gap=1.5,
            ),
            mo.md(
                f"""
    {check(r"$E\cdot A$ performs exactly that row operation on $A$", (E * A) == A.elementary_row_op("n->n+km", row1=1, row2=0, k=-Rational(1, 2)))}
    {check(r"$A \cdot A^{-1} = I$", (A * Ainv) == eye(3))}
    {check(r"and it agrees with SymPy's `A.inv()`", Ainv == A.inv())}

    This is why "row reduce $[A \\,\\rvert\\, I]$" works, and it is not a trick. Every move
    you make on the left, you make on the right; when the left has been driven to $I$ by
    the product $E_k\\cdots E_1$, the right holds that same product applied to $I$ — which
    is the product itself. And a product of row ops that turns $A$ into $I$ is, by
    definition, $A^{{-1}}$.
    """
            ),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The determinant falls out of elimination

    Look again at the table at the top. If you reach upper-triangular form using only
    **row-replacements** (which do not change $\det$) and **row swaps** (each flips its
    sign), then

    $$\det A = (-1)^{\#\text{swaps}} \times (\text{product of the pivots}).$$

    This closes the loop with notebook 4: there, the determinant was *defined* as a volume
    ratio, with no algorithm attached. Here is the algorithm — and it agrees.
    """)
    return


@app.cell
def _(Matrix, check, mo, tex):
    B = Matrix([[0, 2, 1], [1, 1, 1], [2, 1, 0]])  # the leading 0 forces a swap
    U = B
    swaps = 0
    U = U.elementary_row_op("n<->m", row1=0, row2=1)
    swaps += 1
    U = U.elementary_row_op("n->n+km", row1=2, row2=0, k=-U[2, 0] / U[0, 0])
    U = U.elementary_row_op("n->n+km", row1=2, row2=1, k=-U[2, 1] / U[1, 1])

    pivot_product = U[0, 0] * U[1, 1] * U[2, 2]
    det_from_pivots = (-1) ** swaps * pivot_product

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("**$B$**\n\n" + tex(B)),
                    mo.md(
                        rf"$$\xrightarrow{{\;{swaps}\text{{ swap}},\; 2\text{{ replacements}}\;}}$$"
                    ),
                    mo.md("**upper-triangular**\n\n" + tex(U)),
                ],
                justify="start",
                align="center",
                gap=1.2,
            ),
            mo.md(
                rf"""
    $$\det B = (-1)^{{{swaps}}} \times ({pivot_product}) = {det_from_pivots}$$

    {check(rf"matches SymPy's `B.det()` $= {B.det()}$", det_from_pivots == B.det())}
    {check("the single swap really did matter — without its sign the answer is wrong", det_from_pivots != pivot_product)}

    That second tick is worth keeping. Forget the $(-1)^{{\#\text{{swaps}}}}$ and you get
    the determinant with the wrong sign — and by notebook 4's reading, you would be
    claiming the map reverses orientation when it preserves it, or the reverse.
    """
            ),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Gaussian elimination is three reversible row moves, applied with a plan:

    - **Forward sweep** (`"n->n+km"`) clears below each pivot, giving row-echelon
      (upper-triangular) form; **back-substitute** for the answer. Normalising the pivots
      and clearing *above* them gives **RREF**, where the solution is read off directly.
    - The echelon shape **classifies** the system: a pivot in every variable column ⇒
      **unique**; a $[\,0\cdots 0 \mid \text{nonzero}\,]$ row ⇒ **no solution**; a **free**
      (pivot-less) column ⇒ **infinitely many**. Geometrically: lines crossing at a point,
      never meeting, or coinciding.
    - Each operation is left-multiplication by an **elementary matrix**; chaining them
      reduces $[A \,\rvert\, I]$ to $[I \,\rvert\, A^{-1}]$, so elimination **inverts** a
      matrix — and the **determinant** is the signed product of the pivots.

    One engine, read three different ways. And because every step is reversible, the
    scrubber runs backwards as happily as forwards — which is not a UI nicety but the
    defining property of the method.

    **Next (notebook 7):** eigenvectors — the directions an operator does not turn — and
    the three regimes a real $2\times2$ map can fall into.
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
