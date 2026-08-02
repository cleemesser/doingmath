# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mathviz",
#     "wigglystuff>=0.5.23",
#     "sympy>=1.14",
# ]
#
# [tool.uv.sources]
# mathviz = { path = "../mathviz", editable = true }
# ///

# Geometric Linear Algebra 7 -- eigenvectors, as a *reactive* marimo notebook.
# A translation of 07_Eigenvectors_Geometrically.py whose central device is a draggable
# 2x2 matrix: the circle-to-ellipse picture, the exact characteristic polynomial, and the
# regime classification (two real / defective / complex) all recompute as you drag, so the
# three regimes stop being three separate examples and become one continuous family.
#
#   uv run marimo edit GeometricLinearAlgebra/07_Eigenvectors_Geometrically_marimo.py
#
# Draws with `mathviz`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 7 — eigenvectors and eigenvalues, geometrically

    An **eigenvector** is a direction a linear map does not turn. The **eigenvalue** is the
    factor by which it stretches that direction:

    $$A v = \lambda v, \qquad v \neq 0.$$

    Most arrows get turned. The question of this notebook is which ones do not — and the
    surprising answer is that a real $2\times2$ map falls into exactly **three** regimes,
    which you can drag between continuously.

    /// tip | What the marimo version adds
    The jupytext original presents the three regimes as three fixed matrices. Here they are
    one draggable matrix, and the boundaries between regimes are somewhere you can *stand*.
    Park the discriminant near zero and nudge it: watch two eigenvectors slide together,
    merge into one, and then vanish into a complex pair.
    ///
    """)
    return


@app.cell
def _():
    import io

    import marimo as mo
    import numpy as np
    import sympy as sp
    from sympy import Matrix, Rational, simplify, symbols

    import mathviz as mv  # shared plane-viz library (see ../mathviz)
    from wigglystuff import TangleLatex

    return Matrix, Rational, TangleLatex, io, mo, mv, np, simplify, sp, symbols


@app.cell
def _(io, mo, mv, np, sp):
    BLUE, ORANGE, GREEN = mv.BLUE, mv.ORANGE, mv.GREEN
    RED, PURPLE, GREY = mv.RED, mv.PURPLE, mv.GREY

    UNIT_CIRCLE = mv.unit_circle(160)

    def png(scene, width=430):
        """A mathviz scene -> a marimo image (see 01_..._marimo.py for why not .display())."""
        buf = io.BytesIO()
        scene.save(buf)
        return mo.image(buf.getvalue(), width=width)

    def check(claim, ok):
        return f"- {'✅' if ok else '❌'} {claim}"

    def tex(M):
        return f"$${sp.latex(M)}$$"

    def real_eigen(A):
        """Real eigenpairs of a 2x2 array, as a list of (lambda, unit vector).

        numpy always returns a complex pair for a rotation-like map; we keep only the
        genuinely real ones, which is exactly the set of invariant *directions*.
        """
        vals, vecs = np.linalg.eig(np.asarray(A, float))
        out = []
        for i in range(len(vals)):
            if abs(vals[i].imag) < 1e-9 and np.max(np.abs(vecs[:, i].imag)) < 1e-9:
                d = np.real(vecs[:, i])
                n = np.linalg.norm(d)
                if n > 1e-12:
                    out.append((float(vals[i].real), d / n))
        return out

    return BLUE, GREEN, GREY, ORANGE, PURPLE, RED, UNIT_CIRCLE, check, png, real_eigen, tex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The question: which arrows keep their direction?

    Apply a map $A$ to a fan of unit arrows pointing in every direction. Generically each
    output is rotated away from its input. Below, inputs are faint and images bold: the
    **eigen-directions** are exactly the spots where the bold arrow lies *along the same
    line* as the faint one — possibly longer, shorter, or flipped, but not turned off the
    line.

    The default matrix $\begin{pmatrix}2&1\\0&3\end{pmatrix}$ has eigenvalues $2$ and $3$.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo):
    matw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"A = \begin{bmatrix} \tangle{a} & \tangle{b} \\[2pt]"
                r"\tangle{c} & \tangle{d} \end{bmatrix}"
            ),
            parameters={
                "a": {"value": 2.0, "min_value": -4, "max_value": 4, "step": 0.5,
                      "digits": 1, "label": "A, row 1 col 1",
                      "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "b": {"value": 1.0, "min_value": -4, "max_value": 4, "step": 0.5,
                      "digits": 1, "label": "A, row 1 col 2",
                      "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "c": {"value": 0.0, "min_value": -4, "max_value": 4, "step": 0.5,
                      "digits": 1, "label": "A, row 2 col 1",
                      "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "d": {"value": 3.0, "min_value": -4, "max_value": 4, "step": 0.5,
                      "digits": 1, "label": "A, row 2 col 2",
                      "color": {"light": "#147a68", "dark": "#5ed5bd"}},
            },
            editor="inline",
            theme="auto",
        )
    )
    matw
    return (matw,)


@app.cell
def _(Matrix, Rational, matw, np):
    # Steps of 0.5 mean Rational(str(v)) is lossless -- SymPy stays exact all the way to
    # the characteristic polynomial, so eigenvalues come out as radicals, not 1.7320508.
    A = np.array(
        [
            [matw.values["a"], matw.values["b"]],
            [matw.values["c"], matw.values["d"]],
        ]
    )
    As = Matrix(
        [
            [Rational(str(matw.values["a"])), Rational(str(matw.values["b"]))],
            [Rational(str(matw.values["c"])), Rational(str(matw.values["d"]))],
        ]
    )
    return A, As


@app.cell(hide_code=True)
def _(A, BLUE, GREY, ORANGE, PURPLE, UNIT_CIRCLE, check, mo, mv, np, png, real_eigen):
    _eig = real_eigen(A)

    _fan = mv.Plane(extent=4)
    for _ang in np.linspace(0, np.pi, 12, endpoint=False):
        _u = np.array([np.cos(_ang), np.sin(_ang)])
        _fan.vector(_u, color=GREY, alpha=0.5)
        _fan.vector(A @ _u, color=BLUE, alpha=0.9)
    for _lam, _d in _eig:
        _fan.line([0, 0], _d, color=PURPLE)

    _ell = mv.Plane(extent=4)
    _ell.curve(UNIT_CIRCLE, color=GREY, width=2.0, alpha=0.6)
    _ell.curve(UNIT_CIRCLE @ A.T, color=BLUE, width=3.0)
    for _lam, _d, _c in [
        (lam, d, c) for (lam, d), c in zip(_eig, (PURPLE, ORANGE))
    ]:
        _ell.line([0, 0], _d, color=_c)
        _ell.vector(_d, color=_c)
        _ell.vector(_lam * _d, color=_c, alpha=0.6)

    mo.vstack(
        [
            mo.hstack(
                [png(_fan, width=360), png(_ell, width=360)], widths=[1, 1], gap=1.0
            ),
            mo.md(
                rf"""
    **Left:** a fan of inputs (grey) and their images (blue); purple lines mark the
    invariant directions. **Right:** the unit circle becomes an **ellipse**. Along an
    eigenvector the ellipse touches a ray straight out from the origin with no sideways
    swing, and the eigenvalue is how far out it reaches.

    {check("there is at least one real invariant direction", len(_eig) >= 1)}
    {check("there are **two** independent ones", len(_eig) == 2)}
    {check("the map has no real eigenvector at all — every arrow turns", len(_eig) == 0)}

    real eigenvalues found: {", ".join(f"$\\lambda = {lam:+.4f}$" for lam, _ in _eig) if _eig else "*none — they are a complex pair*"}

    Set $c = 0$ and the matrix is triangular, so the eigenvalues are just the diagonal —
    an easy place to start. Then drag $c$ away from zero and watch the eigen-directions
    swing.
    """
            ),
        ],
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The eigenvalue equation and the characteristic polynomial

    $Av = \lambda v$ means $(A - \lambda I)v = 0$ with $v \neq 0$, so $A - \lambda I$ must
    be **singular** — it must crush some nonzero arrow to nothing. By notebook 4 that is
    exactly the statement that its determinant vanishes:

    $$\det(A - \lambda I) = 0.$$

    This is the **characteristic polynomial**. For a $2\times2$ matrix it expands to a form
    worth memorising, because both coefficients are quantities we already understand:

    $$\lambda^2 - (\operatorname{tr}A)\,\lambda + \det A = 0.$$

    SymPy solves it *exactly*, which is the whole pedagogical point — radicals instead of
    `0.7071…`.
    """)
    return


@app.cell
def _(As, check, mo, sp, symbols, tex):
    lam = symbols("lambda")
    charpoly = sp.expand((As - lam * sp.eye(2)).det())
    eigs = As.eigenvals()

    _det, _tr = As.det(), As.trace()
    _disc = sp.simplify(_tr**2 - 4 * _det)

    _lines = []
    for _val, _mult in eigs.items():
        _real = _val.is_real
        _lines.append(
            f"- $\\lambda = {sp.latex(sp.nsimplify(_val))}$ "
            f"(multiplicity {_mult}, {'real' if _real else '**complex**'})"
        )

    # det = product of eigenvalues, tr = sum -- with multiplicity.
    _prod = sp.simplify(sp.prod([v**m for v, m in eigs.items()]))
    _sum = sp.simplify(sum(v * m for v, m in eigs.items()))

    mo.md(
        rf"""
    $$A = {sp.latex(As)} \qquad
      \det(A - \lambda I) = {sp.latex(sp.factor(charpoly))} = 0$$

    $$\operatorname{{tr}} A = {sp.latex(_tr)} \qquad
      \det A = {sp.latex(_det)} \qquad
      \Delta = \operatorname{{tr}}^2 - 4\det = {sp.latex(_disc)}$$

    **eigenvalues**

    {chr(10).join(_lines)}

    {check(r"$\det A = \prod \lambda$ — the volume factor is the product of the stretches", sp.simplify(_prod - _det) == 0)}
    {check(r"$\operatorname{tr} A = \sum \lambda$ — the trace is their sum", sp.simplify(_sum - _tr) == 0)}
    {check(r"the discriminant is positive, so the eigenvalues are **real and distinct**", bool(_disc > 0))}

    Those first two ticks tie this notebook back to notebook 4. The determinant was defined
    there as a volume-scaling factor, with no mention of eigenvalues; the trace as a rate of
    volume change. Here they turn out to be the product and the sum of the stretch factors —
    which is exactly what you would guess if the map were a pure scaling along two axes.
    The content of §3 is that, when it can be, it *is*.
    """
    )
    return (charpoly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Diagonalization: eigen-coordinates make the map a pure scaling

    Stack the eigenvectors as the columns of $P$. In *that* basis the map is just
    independent stretches — the diagonal matrix $D$ of eigenvalues — so

    $$A = P\,D\,P^{-1}.$$

    Geometrically: change to eigen-coordinates, scale each axis by its eigenvalue, change
    back. This is why $A^n$ is easy — the change of basis cancels in the middle:

    $$A^n = P D^n P^{-1}.$$

    It works exactly when there are **enough independent eigenvectors**. When there are
    not, SymPy refuses, and that refusal is the subject of regime (b) below.
    """)
    return


@app.cell
def _(As, check, mo, simplify, sp, tex):
    _n = 5
    try:
        P, D = As.diagonalize()
        _ok = P * D * P.inv() == As
        _pow_ok = simplify(P * D**_n * P.inv() - As**_n) == sp.zeros(2, 2)
        _body = f"""
    $$P = {sp.latex(P)} \\qquad D = {sp.latex(D)}$$

    {check(r"$P D P^{-1} = A$ — the factorization is exact", _ok)}
    {check(rf"$P D^{{{_n}}} P^{{-1}} = A^{{{_n}}}$ — powers are easy in eigen-coordinates", _pow_ok)}

    $$A^{{{_n}}} = {sp.latex(As**_n)}$$

    Read $P$ as "the dictionary between eigen-coordinates and standard coordinates."
    Conjugating by it is the change of basis from notebook 5, and $D$ is what the map
    looks like once you stand in the frame it prefers: two independent stretches, nothing
    more.
    """
    except Exception as _e:
        _body = f"""
    **`As.diagonalize()` raises `{type(_e).__name__}`.**

    {check("this matrix is diagonalizable", False)}

    There are not enough independent eigenvectors to build $P$ — the matrix is
    **defective**. That is not a failure of the method; it is a real geometric fact about
    the map, and regime (b) below is exactly this situation. Drag the entries to restore a
    second independent eigen-direction and the factorization comes back.
    """

    mo.md(_body)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Three regimes a real $2\times2$ map can fall into

    Everything is decided by the **discriminant** $\Delta = \operatorname{tr}^2 - 4\det$ of
    the characteristic polynomial:

    | $\Delta$ | eigenvalues | eigenvectors | example |
    |---|---|---|---|
    | $> 0$ | two distinct reals | two independent | symmetric, or any generic map |
    | $= 0$ | one repeated real | usually only **one** — *defective* | the shear |
    | $< 0$ | a complex pair | **none real** | a rotation |

    Pick a regime to see its canonical example. Then go back and drag the matrix in §1 so
    that $\Delta$ crosses zero — the three rows of this table are a single continuous
    family, not three unrelated species.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    regime = mo.ui.radio(
        options={
            "(a) symmetric ⇒ perpendicular eigen-axes": "sym",
            "(b) shear ⇒ defective, only one eigenvector": "shear",
            "(c) rotation ⇒ complex eigenvalues, none real": "rot",
        },
        value="(a) symmetric ⇒ perpendicular eigen-axes",
        label="regime",
    )
    regime
    return (regime,)


@app.cell(hide_code=True)
def _(
    BLUE,
    GREY,
    Matrix,
    ORANGE,
    PURPLE,
    UNIT_CIRCLE,
    check,
    mo,
    mv,
    np,
    png,
    real_eigen,
    regime,
    sp,
    tex,
):
    if regime.value == "sym":
        _M = Matrix([[2, 1], [1, 2]])
        _title = "### (a) Symmetric ⇒ perpendicular eigen-axes (the spectral theorem)"
        _prose = r"""
    A **symmetric** matrix always has real eigenvalues and **orthogonal** eigenvectors.
    Those are the **principal axes** of the ellipse — the directions of maximum and minimum
    stretch. This is the spectral theorem, and it is the reason principal component
    analysis works: the axes it finds are the eigenvectors of a symmetric covariance matrix,
    and they are guaranteed to be perpendicular.
    """
    elif regime.value == "shear":
        _M = Matrix([[1, 1], [0, 1]])
        _title = "### (b) Shear ⇒ **defective**: a repeated eigenvalue, only one eigenvector"
        _prose = r"""
    A shear fixes the $x$-axis and slides everything parallel to it. Its only invariant
    direction is that axis: $\lambda = 1$ is a **double** root of the characteristic
    polynomial, yet there is just **one** eigenvector. Such a matrix is *defective* and
    **cannot** be diagonalized — there is no basis in which it is a pure scaling.

    This is the unipotent operator of notebook 2, $(H-I)^2 = 0$, seen from the eigenvalue
    side. The algebraic multiplicity is $2$; the geometric multiplicity is $1$; the gap
    between them *is* the defect.
    """
    else:
        _M = Matrix(
            [[sp.Rational(766, 1000), sp.Rational(-643, 1000)],
             [sp.Rational(643, 1000), sp.Rational(766, 1000)]]
        )
        _title = "### (c) Rotation ⇒ **complex** eigenvalues: no real eigenvector at all"
        _prose = r"""
    A pure rotation turns *every* arrow, so it has **no** real eigenvector — the image
    circle coincides with the input circle, yet not one arrow stays on its line. Its
    eigenvalues are the complex pair $e^{\pm i\theta}$.

    This is the $J^2 = -I$ thread from notebooks 2 and 5, one last time: the quarter-turn's
    eigenvalues are $\pm i$, and the scale-and-rotate $xI + yJ$ has eigenvalues $x \pm iy$
    — **the complex number itself**. "No real invariant direction" and "the complex numbers"
    are the same fact wearing different clothes.
    """

    _Mn = np.array(_M.tolist(), dtype=float)
    _eig = real_eigen(_Mn)

    _pl = mv.Plane(extent=4)
    _pl.curve(UNIT_CIRCLE, color=GREY, width=2.0, alpha=0.6)
    _pl.curve(UNIT_CIRCLE @ _Mn.T, color=BLUE, width=3.0)
    for _lam, _d, _c in [(l, d, c) for (l, d), c in zip(_eig, (PURPLE, ORANGE))]:
        _pl.line([0, 0], _d, color=_c)
        _pl.vector(_lam * _d, color=_c)
    for _ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        _u = np.array([np.cos(_ang), np.sin(_ang)])
        _pl.vector(_u, color=GREY, alpha=0.4)
        _pl.vector(_Mn @ _u, color=BLUE, alpha=0.8)

    _disc = float(np.trace(_Mn) ** 2 - 4 * np.linalg.det(_Mn))
    _orth = (
        len(_eig) == 2 and abs(float(np.dot(_eig[0][1], _eig[1][1]))) < 1e-9
    )
    try:
        _M.diagonalize()
        _diagable = True
    except Exception:
        _diagable = False

    mo.vstack(
        [
            mo.md(_title),
            mo.hstack(
                [
                    png(_pl),
                    mo.md(
                        rf"""
    {tex(_M)}

    $$\Delta = \operatorname{{tr}}^2 - 4\det = {_disc:+.4f}
      \qquad \text{{eigenvalues }} {sp.latex(_M.eigenvals())}$$

    {check("has two independent real eigenvectors", len(_eig) == 2)}
    {check("its eigenvectors are **orthogonal**", _orth)}
    {check("it is **diagonalizable**", _diagable)}

    {_prose}
    """
                    ),
                ],
                widths=[1, 1],
                align="center",
                gap=1.2,
            ),
        ],
        gap=0.4,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    An **eigenvector** is a direction a linear map does not turn; the **eigenvalue** is the
    factor it stretches that direction by. Everything else is a consequence:

    | map | eigenvalues | eigenvectors | picture |
    |---|---|---|---|
    | symmetric | $3,\ 1$ (real, distinct) | **orthogonal** | principal axes of the ellipse |
    | shear | $1$ (double) | **one** (defective) | only the $x$-axis is invariant |
    | rotation $R(\theta)$ | $e^{\pm i\theta}$ (complex) | **none real** | every arrow turns |

    - **Find them exactly:** $\det(A-\lambda I) = 0$ for the eigenvalues, then
      $(A-\lambda I)v = 0$ for the eigenvectors. For $2\times2$ this is
      $\lambda^2 - (\operatorname{tr}A)\lambda + \det A = 0$, and the **discriminant**
      $\operatorname{tr}^2 - 4\det$ decides the regime.
    - **Diagonalize:** when there are enough independent eigenvectors, $A = PDP^{-1}$ turns
      the map into independent scalings — making $A^n$, matrix exponentials and stability
      questions trivial.
    - **$\det = \prod\lambda$ and $\operatorname{tr} = \sum\lambda$** — the volume scaling
      of notebook 4 and the trace are just the product and sum of the stretch factors.
    - **The complex thread returns:** a map with no invariant real direction has complex
      eigenvalues; the rotation's are $e^{\pm i\theta}$, and the scale-and-rotate $xI+yJ$
      has eigenvalues $x \pm iy$ — the complex number itself, exactly as notebook 5 found
      it from the other direction.

    The deepest thing on this page is that the three regimes are **one family**. Drag the
    matrix in §1 until the discriminant passes through zero and you will see two
    eigen-directions rush together, collide, and disappear. The defective case is not an
    exception to be memorised; it is the razor-thin boundary between having two real
    directions and having none.
    """)
    return


if __name__ == "__main__":
    app.run()
