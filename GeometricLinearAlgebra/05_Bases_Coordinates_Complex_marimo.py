# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mathviz",
#     "wigglystuff>=0.5.23",
#     "scipy>=1.10",
# ]
#
# [tool.uv.sources]
# mathviz = { path = "../mathviz", editable = true }
# ///

# Geometric Linear Algebra 5 -- coordinates, matrices, and the discovery of C, as a
# *reactive* marimo notebook. A translation of 05_Bases_Coordinates_Complex.py in which the
# basis itself is draggable -- so "coordinates are a report relative to a chosen frame"
# stops being a slogan and becomes something you watch happen.
#
#   uv run marimo edit GeometricLinearAlgebra/05_Bases_Coordinates_Complex_marimo.py
#
# Draws with `mathviz`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 5 — coordinates, matrices, and the discovery of $\mathbb{C}$

    Four notebooks in, we have done geometry with **no coordinates at all**. Operators were
    defined by what they do to arrows; the determinant and trace were defined by what they
    do to volume. Now, at last, we choose a basis — and find that coordinates add no new
    content. They *write down* what we already had.

    Then the payoff. Among all linear operators, the ones that are "a scaling combined with
    a pure rotation" turn out to form a closed, commutative algebra — and that algebra
    **is the complex numbers**. $\mathbb{C}$ is not invented here; it is discovered, as a
    subset of the operators of the plane we have been studying all along.
    """)
    return


@app.cell
def _():
    import io

    import marimo as mo
    import numpy as np
    from scipy.linalg import expm

    import mathviz as mv  # shared plane-viz library (see ../mathviz)
    from wigglystuff import TangleLatex

    return TangleLatex, expm, io, mo, mv, np


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

    def operator_view(M, color, extent=2.5, width=430):
        """Faint domain grid + flag, overlaid with their bold image under the matrix M."""
        p = mv.Plane(extent=extent, grid=False, axes=True)
        p.show_operator(
            M, probe_shape=mv.FLAG, color=color, width=1.8, samples=40,
            probe_shape_color=color,
        )
        return png(p, width=width)

    return BLUE, FAINT, GREEN, GREY, ORANGE, PURPLE, RED, check, operator_view, png


@app.cell
def _(np):
    def J(v):
        """The quarter-turn from notebook 2."""
        v = np.asarray(v, float)
        return np.stack([-v[..., 1], v[..., 0]], axis=-1)

    def coords_in(basis, v):
        """Coordinates of arrow v in `basis` (given as ROWS b1, b2): solve B^T x = v."""
        return np.linalg.solve(np.asarray(basis, float).T, np.asarray(v, float))

    def matrix_of(op):
        """The matrix of an operator = the columns are the images of the basis vectors."""
        return np.array([op(np.array([1.0, 0.0])), op(np.array([0.0, 1.0]))]).T

    def stw(x, y):
        """The scale-and-twist operator xI + yJ, as a 2x2 matrix."""
        return np.array([[x, -y], [y, x]], float)

    def to_complex(M):
        """The dictionary xI + yJ  <->  x + iy, read off the first column."""
        return complex(M[0, 0], M[1, 0])

    return J, coords_in, matrix_of, stw, to_complex


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
    ## 1. A basis turns arrows into coordinates

    Pick two arrows $b_1, b_2$ that are not parallel — a **basis**. Then *every* arrow $v$
    is a unique linear combination

    $$v = x\,b_1 + y\,b_2,$$

    and the pair $(x, y)$ is **the coordinates of $v$ in that basis**.

    Coordinates are not intrinsic to the arrow — they are a *report relative to a chosen
    frame*. The blue arrow below never moves. Drag the purple frame and watch its
    coordinates change anyway. That is the entire content of the word "coordinates," and it
    is why the previous four notebooks were careful never to rely on them.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    basw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"b_1 = \begin{bmatrix} \tangle{b1x} \\[2pt] \tangle{b1y} \end{bmatrix}"
                r" \qquad "
                r"b_2 = \begin{bmatrix} \tangle{b2x} \\[2pt] \tangle{b2y} \end{bmatrix}"
                r"\qquad\text{(the arrow } v \text{ is fixed)}"
            ),
            parameters={
                "b1x": {"value": 1.0, "min_value": -2, "max_value": 2, "step": 0.1,
                        "digits": 1, "label": "b1, x-component",
                        "color": {"light": "#6f5cbd", "dark": "#b9a8ff"}},
                "b1y": {"value": 0.0, "min_value": -2, "max_value": 2, "step": 0.1,
                        "digits": 1, "label": "b1, y-component",
                        "color": {"light": "#6f5cbd", "dark": "#b9a8ff"}},
                "b2x": {"value": 0.6, "min_value": -2, "max_value": 2, "step": 0.1,
                        "digits": 1, "label": "b2, x-component",
                        "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "b2y": {"value": 1.0, "min_value": -2, "max_value": 2, "step": 0.1,
                        "digits": 1, "label": "b2, y-component",
                        "color": {"light": "#147a68", "dark": "#5ed5bd"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    basw
    return (basw,)


@app.cell(hide_code=True)
def _(BLUE, GREEN, GREY, ORANGE, PURPLE, basw, check, coords_in, mo, mv, np, png):
    skew = np.array(
        [
            [basw.values["b1x"], basw.values["b1y"]],
            [basw.values["b2x"], basw.values["b2y"]],
        ]
    )
    varrow = np.array([1.8, 1.2])  # a fixed geometric object
    _degenerate = abs(np.linalg.det(skew)) < 1e-9

    _p = mv.Plane(extent=3.0)
    _p.vector([1, 0], color=GREY, label="e1")
    _p.vector([0, 1], color=GREY, label="e2")
    _p.vector(skew[0], color=PURPLE, label="b1")
    _p.vector(skew[1], color=GREEN, label="b2")
    _p.vector(varrow, color=BLUE, label="v")

    if not _degenerate:
        ck = coords_in(skew, varrow)
        # The orange path walks the two legs: x*b1, then y*b2.
        _p.segment([0, 0], ck[0] * skew[0], color=ORANGE, width=2.0)
        _p.segment(ck[0] * skew[0], varrow, color=ORANGE, width=2.0)
    else:
        ck = np.array([np.nan, np.nan])

    mo.hstack(
        [
            png(_p),
            mo.md(
                rf"""
    the **same** arrow $v = (1.8,\; 1.2)$, reported in two frames:

    $$\text{{standard }} (e_1,e_2): \;(1.800,\; 1.200)
      \qquad
      \text{{yours }} (b_1,b_2): \;({ck[0]:.3f},\; {ck[1]:.3f})$$

    {check(r"the frame is a genuine **basis** — $b_1, b_2$ are not parallel", not _degenerate)}
    {check(r"the coordinates rebuild the original arrow, $x b_1 + y b_2 = v$", (not _degenerate) and np.allclose(ck @ skew, varrow))}
    {check(r"the coordinates are **not** the standard ones — the frame is doing something", not np.allclose(ck, varrow))}

    The orange path walks the two legs $x\,b_1$ then $y\,b_2$ and arrives at the blue
    arrow's tip. Drag $b_1$ and $b_2$ until they line up: the determinant of the frame goes
    to zero, the first tick fails, and the coordinates blow up — because a degenerate
    "basis" cannot reach off its own line, so most arrows have no representation at all and
    the ones that do have infinitely many.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return skew, varrow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The matrix of an operator = where it sends the basis

    A linear operator is pinned down by what it does to the basis (notebook 1). Write each
    image $T(b_j)$ in coordinates and **stack those coordinate-columns** — that array is the
    **matrix of $T$**. Then "apply $T$" becomes "matrix times coordinate-vector," which is
    nothing but *take that linear combination of the columns*:

    $$T(v) = T(x b_1 + y b_2) = x\,T(b_1) + y\,T(b_2).$$

    So the coordinate-free operators of notebook 2 acquire their familiar matrices. Nothing
    new has happened — we have merely started writing them down.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    opw = mo.ui.anywidget(
        TangleLatex(
            latex=r"\text{rotate by } \tangle{ang}^\circ, \qquad \text{shear by } \tangle{k}",
            parameters={
                "ang": {"value": 35, "min_value": -180, "max_value": 180, "step": 5,
                        "digits": 0, "label": "rotation angle (degrees)",
                        "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "k": {"value": 0.8, "min_value": -2, "max_value": 2, "step": 0.1,
                      "digits": 1, "label": "shear strength",
                      "color": {"light": "#b45b1b", "dark": "#ffad66"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    opw
    return (opw,)


@app.cell(hide_code=True)
def _(J, ORANGE, check, matrix_of, mo, np, operator_view, opw):
    _th = np.radians(opw.values["ang"])
    _k = opw.values["k"]

    def _rotate(z):
        return np.cos(_th) * np.asarray(z, float) + np.sin(_th) * J(z)

    def _shear(z):
        z = np.asarray(z, float)
        return z + _k * z[..., 1, None] * np.array([1.0, 0.0])

    # Built from the geometry: columns = images of e1, e2. NOT written down by hand.
    Mrot = matrix_of(_rotate)
    Mshear = matrix_of(_shear)
    _closed_rot = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
    _closed_shear = np.array([[1.0, _k], [0.0, 1.0]])

    _test = np.random.default_rng(5).normal(size=(4, 2))

    mo.hstack(
        [
            operator_view(Mshear, ORANGE),
            mo.md(
                rf"""
    $$M_{{\text{{rot}}}} = \begin{{bmatrix}} {Mrot[0, 0]:+.3f} & {Mrot[0, 1]:+.3f} \\
      {Mrot[1, 0]:+.3f} & {Mrot[1, 1]:+.3f} \end{{bmatrix}}
      \qquad
      M_{{\text{{shear}}}} = \begin{{bmatrix}} {Mshear[0, 0]:+.3f} & {Mshear[0, 1]:+.3f} \\
      {Mshear[1, 0]:+.3f} & {Mshear[1, 1]:+.3f} \end{{bmatrix}}$$

    {check(r"the rotation matrix, read off the basis images, matches the closed form $\begin{bmatrix}\cos&-\sin\\\sin&\cos\end{bmatrix}$", np.allclose(Mrot, _closed_rot))}
    {check(r"so does the shear's, $\begin{bmatrix}1&k\\0&1\end{bmatrix}$", np.allclose(Mshear, _closed_shear))}
    {check(r"and **matrix·v reproduces the geometric action** on random arrows", np.allclose(_test @ Mrot.T, _rotate(_test)))}

    Both matrices were *computed* — `matrix_of` applies the geometric operator to $e_1$ and
    $e_2$ and stacks the answers. They were not typed in. That they come out equal to the
    formulas in every textbook is the point: the formulas were always just a record of
    where the basis went.

    The picture is the shear acting on coordinates — and it is pixel-for-pixel the same
    picture as notebook 2, where no matrix existed.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return (Mrot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Determinant and trace, recovered — and basis-independent

    With a matrix $\begin{pmatrix}a&b\\c&d\end{pmatrix}$ in hand, the coordinate-free
    definitions collapse to the formulas everyone memorises:

    $$\det = ad - bc \quad(\text{the wedge of the columns}), \qquad \operatorname{tr} = a + d.$$

    Now the crucial check. The *value* must not depend on the basis we happened to choose,
    or the notebook-4 definitions — which never mentioned a basis — would have been
    nonsense. Re-expressing an operator in another frame conjugates its matrix,
    $A \mapsto P^{-1}AP$, and both $\det$ and $\operatorname{tr}$ must be **invariant**.
    """)
    return


@app.cell
def _(check, mo, np, skew):
    A = np.array([[1.3, -0.7], [0.4, 1.1]])
    _det_formula = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    _tr_formula = A[0, 0] + A[1, 1]

    # Invariance under a *random* change of basis, 1000 times...
    _rng = np.random.default_rng(6)
    _inv_ok = True
    for _ in range(1000):
        P = _rng.normal(size=(2, 2))
        if abs(np.linalg.det(P)) < 1e-6:
            continue
        B = np.linalg.inv(P) @ A @ P  # the same operator, a different basis
        _inv_ok &= np.allclose(np.linalg.det(B), np.linalg.det(A)) and np.allclose(
            np.trace(B), np.trace(A)
        )

    # ...and under the specific frame you dragged in section 1.
    _yours_ok = False
    if abs(np.linalg.det(skew)) > 1e-9:
        _Byours = np.linalg.inv(skew.T) @ A @ skew.T
        _yours_ok = np.allclose(np.linalg.det(_Byours), np.linalg.det(A)) and np.allclose(
            np.trace(_Byours), np.trace(A)
        )

    mo.md(
        rf"""
    $$A = \begin{{bmatrix}} 1.3 & -0.7 \\ 0.4 & 1.1 \end{{bmatrix}}
      \qquad ad - bc = {_det_formula:.4f} \qquad a + d = {_tr_formula:.4f}$$

    {check(r"$ad-bc$ agrees with `np.linalg.det`", np.allclose(_det_formula, np.linalg.det(A)))}
    {check(r"$a+d$ agrees with `np.trace`", np.allclose(_tr_formula, np.trace(A)))}
    {check(r"$\det$ and $\operatorname{tr}$ survive **1000 random changes of basis**, $A\mapsto P^{-1}AP$", _inv_ok)}
    {check(r"…including the frame you dragged in §1", _yours_ok)}

    This is the retroactive justification for notebooks 3 and 4. We defined the determinant
    as a volume ratio and the trace as a rate of volume change, never mentioning
    coordinates. Had those quantities turned out to depend on the frame, the definitions
    would have been incoherent. They do not — so "the determinant of an operator" is a
    legitimate thing to say, and $ad-bc$ is merely how you compute it once you have picked
    somewhere to stand.
    """
    )
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The discovery of $\mathbb{C}$: scale-and-twist operators

    Now the destination. Among all linear operators, single out the **scale-and-twist**:
    scale every arrow by some $r \ge 0$ *and* rotate it by some angle $\phi$. From notebook
    2, rotation is $\cos\phi\,I + \sin\phi\,J$, so a scale-and-twist is

    $$r(\cos\phi\,I + \sin\phi\,J) = x\,I + y\,J, \qquad x = r\cos\phi,\; y = r\sin\phi.$$

    So the scale-and-twists are exactly the **real combinations $xI + yJ$** of the identity
    and the quarter-turn:

    $$xI + yJ = \begin{pmatrix} x & -y \\ y & x\end{pmatrix}.$$

    Watch this set of operators. It is a **2-dimensional plane** spanned by $I$ and $J$,
    and — the key fact — it is **closed under composition**, because the only product we
    ever need is $J^2 = -I$. Unlike operators in general, these also **commute**.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    zw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"z = \tangle{zx}\,I + \tangle{zy}\,J"
                r" \qquad\qquad "
                r"w = \tangle{wx}\,I + \tangle{wy}\,J"
            ),
            parameters={
                "zx": {"value": 1.3, "min_value": -2, "max_value": 2, "step": 0.1,
                       "digits": 1, "label": "z, real part",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "zy": {"value": 0.9, "min_value": -2, "max_value": 2, "step": 0.1,
                       "digits": 1, "label": "z, imaginary part",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "wx": {"value": 0.5, "min_value": -2, "max_value": 2, "step": 0.1,
                       "digits": 1, "label": "w, real part",
                       "color": {"light": "#b45b1b", "dark": "#ffad66"}},
                "wy": {"value": 1.4, "min_value": -2, "max_value": 2, "step": 0.1,
                       "digits": 1, "label": "w, imaginary part",
                       "color": {"light": "#b45b1b", "dark": "#ffad66"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    zw
    return (zw,)


@app.cell
def _(check, mo, np, stw, to_complex, zw):
    zvec = np.array([zw.values["zx"], zw.values["zy"]])
    wvec = np.array([zw.values["wx"], zw.values["wy"]])
    Mz, Mw = stw(*zvec), stw(*wvec)
    I2, Jm = stw(1, 0), stw(0, 1)

    _prod = Mz @ Mw
    _x_out, _y_out = _prod[0, 0], _prod[1, 0]
    _rule = [
        zvec[0] * wvec[0] - zvec[1] * wvec[1],
        zvec[0] * wvec[1] + wvec[0] * zvec[1],
    ]

    # The dictionary must respect BOTH operations, on random elements as well.
    _rng = np.random.default_rng(3)
    _iso = True
    for _ in range(2000):
        _a, _b = _rng.normal(size=2), _rng.normal(size=2)
        _M1, _M2 = stw(*_a), stw(*_b)
        _iso &= np.isclose(to_complex(_M1 + _M2), to_complex(_M1) + to_complex(_M2))
        _iso &= np.isclose(to_complex(_M1 @ _M2), to_complex(_M1) * to_complex(_M2))

    mo.md(
        rf"""
    $$z\,w \;=\; \begin{{bmatrix}} {_prod[0, 0]:+.3f} & {_prod[0, 1]:+.3f} \\
      {_prod[1, 0]:+.3f} & {_prod[1, 1]:+.3f} \end{{bmatrix}}
      \;=\; ({_x_out:+.3f})\,I + ({_y_out:+.3f})\,J$$

    {check(r"$J^2 = -I$ — the seed of everything", np.allclose(Jm @ Jm, -I2))}
    {check(r"**closed**: the product of two scale-and-twists is again a scale-and-twist", np.allclose(_prod, stw(_x_out, _y_out)))}
    {check(r"**commutative**, $zw = wz$ — very much not true of operators in general", np.allclose(Mz @ Mw, Mw @ Mz))}
    {check(r"the composed coordinates follow $(x_1x_2 - y_1y_2,\; x_1y_2 + x_2y_1)$", np.allclose([_x_out, _y_out], _rule))}
    {check(r"the dictionary $xI+yJ \leftrightarrow x+iy$ respects **addition and multiplication**, on 2000 random pairs", _iso)}

    That composition rule is **exactly complex multiplication**. The dictionary

    $$x\,I + y\,J \;\longleftrightarrow\; x + i\,y$$

    is an isomorphism: addition ↔ addition, composition ↔ multiplication,
    $J \leftrightarrow i$, and $J^2 = -I \leftrightarrow i^2 = -1$.
    **The complex numbers are the algebra of scale-and-twist operators of the plane.**
    """
    )
    return I2, Jm, Mw, Mz, wvec, zvec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modulus, argument, determinant, trace — all geometric

    Reading $z = xI + yJ$ as the operator "scale by $r$, rotate by $\phi$", every piece of
    complex-number vocabulary turns out to be a geometric statement we already made:

    - **modulus** $\lvert z\rvert = r = \sqrt{x^2+y^2}$ is the **length-scaling** factor;
    - **argument** $\arg z = \phi$ is the **rotation angle**;
    - multiplying complex numbers **multiplies moduli and adds arguments** — because
      composing scale-and-twists multiplies the scales and adds the angles;
    - the **determinant** is $x^2+y^2 = \lvert z\rvert^2$, the **area**-scaling factor of
      notebook 4 — and indeed scaling lengths by $r$ scales areas by $r^2$;
    - the **trace** is $2x = 2\,\mathrm{Re}(z)$.

    And the capstone — **Euler's formula, as a theorem about operators**:

    $$e^{\phi J} = \cos\phi\,I + \sin\phi\,J \;\longleftrightarrow\; e^{i\phi} = \cos\phi + i\sin\phi.$$
    """)
    return


@app.cell(hide_code=True)
def _(GREEN, Jm, Mw, Mz, check, expm, mo, np, operator_view, wvec, zvec):
    r = float(np.hypot(*zvec))
    phi = float(np.arctan2(zvec[1], zvec[0]))
    _zw = Mz @ Mw
    _phi0 = 0.7
    _R = np.array(
        [[np.cos(_phi0), -np.sin(_phi0)], [np.sin(_phi0), np.cos(_phi0)]]
    )

    mo.hstack(
        [
            operator_view(Mz, GREEN),
            mo.md(
                rf"""
    $$z = {zvec[0]:+.1f} {zvec[1]:+.1f}i
      \qquad \lvert z\rvert = {r:.4f}
      \qquad \arg z = {np.degrees(phi):.2f}^\circ$$

    {check(rf"$\det = \lvert z\rvert^2 = {r ** 2:.4f}$ — the **area** factor", np.allclose(np.linalg.det(Mz), r ** 2))}
    {check(rf"$\operatorname{{tr}} = 2\,\mathrm{{Re}}\,z = {2 * zvec[0]:.2f}$", np.allclose(np.trace(Mz), 2 * zvec[0]))}
    {check(r"$\lvert zw\rvert = \lvert z\rvert\,\lvert w\rvert$ — **moduli multiply**", np.allclose(np.hypot(_zw[0, 0], _zw[1, 0]), np.hypot(*zvec) * np.hypot(*wvec)))}
    {check(r"$\arg(zw) = \arg z + \arg w$ — **arguments add**", np.allclose(np.exp(1j * np.arctan2(_zw[1, 0], _zw[0, 0])), np.exp(1j * (np.arctan2(zvec[1], zvec[0]) + np.arctan2(wvec[1], wvec[0])))))}
    {check(r"**Euler**: $e^{\phi J}$ *is* the rotation matrix $R(\phi)$", np.allclose(expm(_phi0 * Jm), _R))}

    The picture is multiply-by-$z$ acting on the plane: a scale by
    $\lvert z\rvert = {r:.3f}$ and a rotation by ${np.degrees(phi):.1f}^\circ$, together, in
    one complex multiplication. The flag keeps its shape and its handedness — scale-and-twists
    are exactly the linear maps that do.

    The argument tick is compared as $e^{{i\arg}}$ rather than as a raw angle on purpose:
    arguments add only up to multiples of $2\pi$, so drag $z$ and $w$ far enough round and a
    naive angle comparison would report a spurious failure where the geometry is perfectly
    fine.
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
    ## Summary — and the arc of the series

    - A **basis** assigns each arrow **coordinates**; they are a frame-dependent report, not
      intrinsic. Drag the frame, the numbers change, the arrow does not.
    - The **matrix** of an operator stacks the coordinates of the basis images;
      matrix·vector is just "that linear combination of the columns." Matrices add no new
      content — they *write down* the operators of notebook 2.
    - **Determinant** $= ad-bc$ (the wedge of the columns) and **trace** $= a+d$ recover the
      coordinate-free definitions of notebook 4, and are **basis-independent** under
      $A\mapsto P^{-1}AP$ — which is what made those definitions well-posed in the first
      place.
    - The operators that **scale and twist** are precisely $xI+yJ$, closed and commutative
      under composition because $J^2 = -I$. That algebra **is $\mathbb{C}$**:
      $z = x+iy \leftrightarrow xI+yJ$, with $\lvert z\rvert$ = length scaling, $\arg z$ =
      rotation, $\lvert z\rvert^2 = \det$ = area scaling, and $e^{i\phi} = e^{\phi J}$ =
      rotation. **Complex numbers are the scale-and-twists of the plane.**

    **The whole arc.** We began with arrows and two operations (nb 1), defined the operators
    of geometry without coordinates (nb 2), built the dot and wedge to measure length,
    angle, area and volume (nb 3), distilled volume into the determinant and its
    infinitesimal shadow the trace (nb 4), and only at the end introduced coordinates —
    discovering that they merely re-encode what we already had, and that the plane's own
    scale-and-twist operators are the complex numbers (nb 5).

    For the same secret approached from the side of *groups* rather than *operators* —
    $SO(2) \cong U(1)$, $\mathbb{C}^\times$, $SU(2)$ and beyond — continue with the
    [`LieGroups/`](../LieGroups/README.md) series. For elimination, eigenvectors and the
    dual basis, carry on with notebooks 6, 7 and 9.
    """)
    return


if __name__ == "__main__":
    app.run()
