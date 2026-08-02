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

# Geometric Linear Algebra 4 -- volume, determinant, trace, as a *reactive* marimo notebook.
# A translation of 04_Volume_Determinant_Trace.py built around one draggable operator whose
# three stretch factors and one shear drive the determinant, the trace, and the exponential
# bridge between them.
#
#   uv run marimo edit GeometricLinearAlgebra/04_Volume_Determinant_Trace_marimo.py
#
# Draws with `mathviz`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 4 — volume, the determinant, and a coordinate-free trace

    Notebook 3 ended on a promise: an operator multiplies *every* signed area by one
    universal factor. Now we name that factor, extend it to volume in any dimension, and
    then differentiate it.

    Two numbers come out, and they are the same number seen at two time-scales:

    - the **determinant** — the *finite* factor by which $T$ scales signed volume;
    - the **trace** — its *infinitesimal* shadow, the rate at which volume starts to change
      when you nudge the identity in the direction of $T$.

    Neither definition mentions a basis. "Product of the eigenvalues" and "sum of the
    diagonal" are *consequences* we recover in notebook 5, not starting points.
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

    def space(bounds=3.0, **kw):
        """A 3D scene on the matplotlib backend -- ~3x faster than vedo, which matters
        when the scene re-renders on every drag."""
        return mv.Space3D(bounds=bounds, backend="mpl", **kw)

    return BLUE, FAINT, GREEN, GREY, ORANGE, PURPLE, RED, check, png, space


@app.cell
def _(np):
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
        return float(np.dot(np.asarray(u, float), cross3(v, w)))

    def det_by_volume(A):
        """The determinant as a *volume ratio*: the wedge of the columns over the unit box.

        Note this never calls np.linalg.det -- it is the coordinate-free definition,
        computed, so that agreeing with numpy downstream is a real check and not a tautology.
        """
        A = np.asarray(A, float)
        cols = [A[:, i] for i in range(A.shape[0])]
        if A.shape[0] == 2:
            return cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0]
        return wedge3(*cols)

    def trace_by_derivative(A, t=1e-6):
        """tr A as the t -> 0 rate of change of det(I + tA)."""
        A = np.asarray(A, float)
        return (np.linalg.det(np.eye(A.shape[0]) + t * A) - 1.0) / t

    def trace_by_wedge(A):
        """The basis-free trace: replace one edge at a time by its image, and add up."""
        A = np.asarray(A, float)
        u, v, w = np.eye(3)
        return (
            wedge3(A @ u, v, w) + wedge3(u, A @ v, w) + wedge3(u, v, A @ w)
        ) / wedge3(u, v, w)

    def divergence(F, x, h=1e-6):
        """Divergence of a vector field by central differences."""
        x = np.asarray(x, float)
        total = 0.0
        for i in range(x.size):
            e = np.zeros(x.size)
            e[i] = h
            total += (F(x + e)[i] - F(x - e)[i]) / (2 * h)
        return total

    return (
        cross3,
        det_by_volume,
        divergence,
        trace_by_derivative,
        trace_by_wedge,
        wedge3,
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
    ## 1. The volume element: the top-degree wedge

    The wedge keeps going. Wedging the largest number of arrows the space allows gives a
    single signed number — the **volume element**:

    - **Plane (2 arrows):** $u\wedge v$ = signed **area** (notebook 3).
    - **Space (3 arrows):** $u\wedge v\wedge w$ = signed **volume** of the parallelepiped
      they span.

    Concretely the 3D volume element is the **scalar triple product** $u\cdot(v\times w)$.
    It inherits the wedge's personality exactly: **multilinear** (linear in each arrow) and
    **alternating** (it vanishes whenever two arrows coincide — a flat, volumeless box), so
    swapping any two arrows flips the sign. The sign is the 3D orientation: right-handed
    $+$, left-handed $-$.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    boxw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"w = \begin{bmatrix} \tangle{wx} \\[2pt] \tangle{wy} \\[2pt] \tangle{wz} \end{bmatrix}"
                r"\qquad u = \begin{bmatrix} 2.0 \\ 0.3 \\ 0.2 \end{bmatrix}"
                r"\qquad v = \begin{bmatrix} 0.4 \\ 1.8 \\ 0.1 \end{bmatrix}"
            ),
            parameters={
                "wx": {"value": 0.2, "min_value": -2.5, "max_value": 2.5, "step": 0.1,
                       "digits": 1, "label": "w, x-component",
                       "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "wy": {"value": 0.5, "min_value": -2.5, "max_value": 2.5, "step": 0.1,
                       "digits": 1, "label": "w, y-component",
                       "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "wz": {"value": 1.6, "min_value": -2.5, "max_value": 2.5, "step": 0.1,
                       "digits": 1, "label": "w, z-component",
                       "color": {"light": "#147a68", "dark": "#5ed5bd"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    boxw
    return (boxw,)


@app.cell(hide_code=True)
def _(BLUE, GREEN, ORANGE, boxw, check, mo, np, png, space, wedge3):
    ubox = np.array([2.0, 0.3, 0.2])
    vbox = np.array([0.4, 1.8, 0.1])
    wbox = np.array([boxw.values["wx"], boxw.values["wy"], boxw.values["wz"]])
    _vol = wedge3(ubox, vbox, wbox)

    _s = space(bounds=3.0)
    _s.parallelepiped([0, 0, 0], ubox, vbox, wbox, facecolor=BLUE, alpha=0.3)
    _s.arrow([0, 0, 0], ubox, color=BLUE)
    _s.arrow([0, 0, 0], vbox, color=ORANGE)
    _s.arrow([0, 0, 0], wbox, color=GREEN)

    mo.hstack(
        [
            png(_s),
            mo.md(
                rf"""
    $$u\wedge v\wedge w \;=\; u\cdot(v\times w) \;=\; {_vol:+.4f}$$

    {check(r"**alternating**: $u\wedge u\wedge w = 0$ — a box with a repeated edge is flat", np.isclose(wedge3(ubox, ubox, wbox), 0.0))}
    {check(r"swapping two edges flips the sign", np.isclose(_vol, -wedge3(vbox, ubox, wbox)))}
    {check(r"**multilinear**: scaling one edge by $3$ scales the volume by $3$", np.isclose(wedge3(3 * ubox, vbox, wbox), 3 * _vol))}
    {check(r"adding a multiple of $u$ to $w$ changes nothing — shearing a box preserves volume", np.isclose(wedge3(ubox, vbox, wbox + 2.5 * ubox), _vol))}
    {check(r"the frame is **right-handed** (positive volume)", _vol > 0)}
    {check(r"the three edges are **coplanar** — the box has collapsed", np.isclose(_vol, 0.0))}

    That fourth tick is the alternating property doing real work: $w$ and $w + 2.5u$ span
    the same box because the extra $u$ slides the top face parallel to the base. Base times
    height, unchanged. It is also why the determinant will be blind to shears.

    Drag $w$ into the plane of $u$ and $v$ and the volume passes through $0$, then comes
    back **negative** — the frame has turned left-handed. Nothing about the box's *size*
    changed discontinuously; only its orientation flipped.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return ubox, vbox, wbox


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The determinant, defined by volume

    Apply an operator $T$ to every edge of the volume element. Multilinearity plus the
    alternating property force the result to be the *same* element times a single scalar,
    no matter which edges you started with:

    $$T(u)\wedge T(v)\wedge T(w) = (\det T)\;(u\wedge v\wedge w).$$

    That scalar **is** the determinant — defined coordinate-free as the factor by which $T$
    scales signed volume. Everything you know about determinants now reads off as geometry:

    - $\lvert\det T\rvert$ is the **volume-scaling factor**; its **sign** says whether $T$
      preserves or flips orientation.
    - $\det T = 0 \iff T$ **collapses** the volume element into a lower dimension, so $T$ is
      not invertible.
    - $\det(T\circ S) = \det T \cdot \det S$ — scaling volume by $S$ and then by $T$
      multiplies the factors.

    The operator below stretches by $s_x, s_y, s_z$ along the three axes and then shears by
    $k$. Drag $k$ as far as you like: **the determinant will not move.**
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    opw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"T = \begin{bmatrix}"
                r"\tangle{sx} & \tangle{k} & 0 \\[2pt]"
                r"0 & \tangle{sy} & 0 \\[2pt]"
                r"0 & 0 & \tangle{sz}"
                r"\end{bmatrix}"
            ),
            parameters={
                "sx": {"value": 1.4, "min_value": -2, "max_value": 2.5, "step": 0.1,
                       "digits": 1, "label": "stretch along x",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "sy": {"value": 1.1, "min_value": -2, "max_value": 2.5, "step": 0.1,
                       "digits": 1, "label": "stretch along y",
                       "color": {"light": "#b45b1b", "dark": "#ffad66"}},
                "sz": {"value": 1.0, "min_value": -2, "max_value": 2.5, "step": 0.1,
                       "digits": 1, "label": "stretch along z",
                       "color": {"light": "#147a68", "dark": "#5ed5bd"}},
                "k": {"value": 0.0, "min_value": -2, "max_value": 2, "step": 0.1,
                      "digits": 1, "label": "shear (does not change the volume)",
                      "color": {"light": "#6f5cbd", "dark": "#b9a8ff"}},
            },
            editor="inline",
            theme=theme.value,
        )
    )
    opw
    return (opw,)


@app.cell
def _(np, opw):
    # One operator drives sections 2, 4 and 5.
    T3 = np.array(
        [
            [opw.values["sx"], opw.values["k"], 0.0],
            [0.0, opw.values["sy"], 0.0],
            [0.0, 0.0, opw.values["sz"]],
        ]
    )
    return (T3,)


@app.cell(hide_code=True)
def _(BLUE, FAINT, T3, check, det_by_volume, mo, np, png, space, wedge3):
    _e1, _e2, _e3 = np.eye(3)
    _det = det_by_volume(T3)

    _s = space(bounds=2.5)
    _s.parallelepiped([0, 0, 0], _e1, _e2, _e3, facecolor=FAINT, alpha=0.12)
    _s.parallelepiped(
        [0, 0, 0], T3 @ _e1, T3 @ _e2, T3 @ _e3, facecolor=BLUE, alpha=0.3
    )

    # The factor must not depend on which edges we picked -- test on unrelated random ones.
    _rng = np.random.default_rng(4)
    _factors = []
    for _ in range(5):
        _a, _b, _c = _rng.normal(size=3), _rng.normal(size=3), _rng.normal(size=3)
        _base = wedge3(_a, _b, _c)
        if abs(_base) > 1e-9:
            _factors.append(wedge3(T3 @ _a, T3 @ _b, T3 @ _c) / _base)

    _A = _rng.normal(size=(3, 3))

    mo.hstack(
        [
            png(_s),
            mo.md(
                rf"""
    $$\det T = {_det:+.4f}
      \qquad \text{{unit cube (volume }}1\text{{)}} \;\mapsto\;
      \text{{box of volume }} {abs(_det):.4f}$$

    {check(r"the volume factor is the **same for five unrelated random frames**", np.allclose(_factors, _det))}
    {check(r"the coordinate-free definition agrees with `np.linalg.det`", np.allclose(_det, np.linalg.det(T3)))}
    {check(r"**multiplicative**, $\det(TA) = \det T \cdot \det A$", np.allclose(det_by_volume(T3 @ _A), det_by_volume(T3) * det_by_volume(_A)))}
    {check(r"$T$ preserves volume ($\det T = 1$)", np.isclose(_det, 1.0))}
    {check(r"$T$ preserves orientation ($\det T > 0$)", _det > 0)}
    {check(r"$T$ is **invertible** ($\det T \neq 0$)", not np.isclose(_det, 0.0))}

    The faint wireframe is the unit cube; the blue box is its image. Drag the shear $k$ —
    the box visibly leans, and $\det T$ does not change by a hair. Drag any stretch to
    $0$ — the box flattens to a rectangle, $\det T$ hits $0$, and the invertibility tick
    goes red. There is no way to un-flatten it, which is exactly what "not invertible"
    means, stated as a picture.
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
    ### The notebook-2 operators, in the plane

    A quick tour confirming the geometry: the operators we defined coordinate-free in
    notebook 2 have exactly the determinants their pictures demanded.
    """)
    return


@app.cell
def _(det_by_volume, mo, np):
    _th = 0.6
    _ops2d = {
        "rotation θ=0.6": np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]]),
        "uniform scale 1.6": 1.6 * np.eye(2),
        "shear k=0.8": np.array([[1.0, 0.8], [0.0, 1.0]]),
        "projection onto x": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "reflection in x": np.array([[1.0, 0.0], [0.0, -1.0]]),
    }

    def _row(name, M):
        d = det_by_volume(M)
        tag = {
            1: "preserves area and orientation",
            -1: "**flips** orientation",
            0: "**collapses** to a line",
        }.get(round(d), f"scales area by {abs(d):.2f}")
        return f"| {name} | {d:+.3f} | {tag} |"

    mo.md(
        "\n".join(
            ["| operator | det | geometric reading |", "|---|---|---|"]
            + [_row(n, M) for n, M in _ops2d.items()]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The trace: the determinant's infinitesimal shadow

    The determinant is the *finite* volume factor. Its **first-order** version is the
    **trace**. Nudge the identity a little in the direction of $T$ and ask how fast volume
    starts to change:

    $$\operatorname{tr} T \;=\; \left.\frac{d}{dt}\right|_{t=0} \det(I + tT)$$

    This is a coordinate-free definition — it never mentions a basis, only volumes.
    Differentiating the volume-element definition by the product rule gives an equally
    basis-free *formula*: replace one edge at a time by its $T$-image, and add up.

    $$\operatorname{tr} T = \frac{(Tu)\wedge v\wedge w \;+\; u\wedge(Tv)\wedge w \;+\; u\wedge v\wedge(Tw)}{u\wedge v\wedge w}$$

    Geometrically: the trace is the **total first-order stretch summed over independent
    directions**. Notice that the shear $k$ sits off the diagonal and contributes nothing
    to it — drag $k$ and the trace, like the determinant, ignores you.
    """)
    return


@app.cell
def _(T3, check, mo, np, trace_by_derivative, trace_by_wedge):
    tr_deriv = trace_by_derivative(T3)
    tr_wedge = trace_by_wedge(T3)
    tr_diag = np.trace(T3)

    mo.md(
        rf"""
    | route to the trace | value |
    |---|---|
    | $\frac{{d}}{{dt}}\big\rvert_0 \det(I + tT)$ — the definition, by finite difference | ${tr_deriv:+.5f}$ |
    | the wedge formula — basis-free, one edge replaced at a time | ${tr_wedge:+.5f}$ |
    | `np.trace` — sum of the diagonal, the *coordinate* recipe | ${tr_diag:+.5f}$ |

    {check("all three agree", np.allclose(tr_deriv, tr_diag, atol=1e-4) and np.allclose(tr_wedge, tr_diag))}

    The third row is the one every textbook starts with, and it is the only one that looks
    like it depends on a basis. It does not — notebook 5 proves $\operatorname{{tr}}$ is
    unchanged by $A \mapsto P^{{-1}}AP$. The first two rows are why: they never mention
    coordinates at all.
    """
    )
    return (tr_diag,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The trace is the divergence of the linear flow

    Read $T$ as a **velocity field** $F(x) = Tx$: at each point, move with velocity $Tx$.
    The divergence $\nabla\!\cdot\!F$ measures the local rate of volume expansion of the
    flow — and for a linear field it is constant everywhere and equal to
    $\operatorname{tr}T$. A little ball of points carried by the flow inflates (or
    deflates) at the fractional rate $\operatorname{tr}T$ per unit time.

    The arrows below *are* the field $x \mapsto Tx$. Where the trace is positive the arrows
    point outward on balance and volume grows; make it negative and the flow drains inward.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, T3, check, divergence, mo, mv, np, png, tr_diag):
    _rng = np.random.default_rng(11)
    _pts = _rng.normal(size=(5, 3))
    _divs = [divergence(lambda x: T3 @ x, _x) for _x in _pts]

    # The field in the z = 0 slice, where it is a genuinely 2D picture.
    _T2 = T3[:2, :2]
    _p = mv.Plane(extent=2.5, grid=False)
    _p.field(_T2, n=15, color=BLUE)

    mo.hstack(
        [
            png(_p),
            mo.md(
                rf"""
    divergence of $x\mapsto Tx$, sampled at five unrelated points:

    $$[\;{", ".join(f"{d:+.4f}" for d in _divs)}\;]$$

    {check(r"the divergence is the **same at every point** — a linear flow expands uniformly", np.allclose(_divs, _divs[0]))}
    {check(rf"and it equals $\operatorname{{tr}} T = {tr_diag:+.4f}$", np.allclose(_divs, tr_diag))}
    {check(r"the flow is **volume-preserving** ($\operatorname{tr} T = 0$)", np.isclose(tr_diag, 0.0))}
    {check(r"the flow is **expanding** ($\operatorname{tr} T > 0$)", tr_diag > 0)}

    The picture is the $z=0$ slice of the field (colour encodes speed). Drag the stretches
    until the trace passes through zero and the flow changes character: what was a source
    becomes a sink. That single number — the trace — is the whole story of whether volume
    grows or shrinks, and it is *blind* to the shear that so obviously distorts the flow.
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
    ### The bridge: $\det e^{tT} = e^{\,t\operatorname{tr}T}$

    The trace is a *rate*; the determinant is a *total*. Integrating the rate
    interpretation over time gives the clean exponential law connecting them:

    $$\det\!\big(e^{tT}\big) = e^{\,t\,\operatorname{tr}T}.$$

    Volume under the flow grows exactly exponentially, with rate the trace — the finite
    determinant is the time-integral of the infinitesimal trace. This is the identity
    behind "$\mathfrak{sl}_n$ = traceless $\iff$ $SL_n$ = volume-preserving" in the
    `LieGroups/` notebooks, and it is why the trace is the right notion of "infinitesimal
    determinant" rather than some other first-order quantity.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo, theme):
    tw = mo.ui.anywidget(
        TangleLatex(
            latex=r"\det\!\big(e^{\tangle{t} T}\big) \;=\; e^{\,\tangle{t}\,\operatorname{tr}T}",
            parameters={
                "t": {"value": 1.0, "min_value": -3, "max_value": 3, "step": 0.1,
                      "digits": 1, "label": "time along the flow",
                      "color": {"light": "#a03050", "dark": "#ff8fa8"}},
            },
            reveal_all_on_drag=True,
            editor="inline",
            theme=theme.value,
        )
    )
    tw
    return (tw,)


@app.cell
def _(T3, check, expm, mo, np, tr_diag, tw):
    tflow = tw.values["t"]
    _lhs = float(np.linalg.det(expm(tflow * T3)))
    _rhs = float(np.exp(tflow * tr_diag))

    _rngb = np.random.default_rng(5)
    _A, _B = _rngb.normal(size=(3, 3)), _rngb.normal(size=(3, 3))

    mo.md(
        rf"""
    $$\det\!\big(e^{{{tflow:.1f}T}}\big) = {_lhs:.6f}
      \qquad e^{{\,{tflow:.1f}\times{tr_diag:.4f}}} = {_rhs:.6f}$$

    {check(rf"the two sides agree at $t = {tflow:.1f}$", np.allclose(_lhs, _rhs))}
    {check(r"the trace is **linear**, $\operatorname{tr}(2A - 3B) = 2\operatorname{tr}A - 3\operatorname{tr}B$", np.allclose(np.trace(2 * _A - 3 * _B), 2 * np.trace(_A) - 3 * np.trace(_B)))}
    {check(r"the trace is **cyclic**, $\operatorname{tr}(AB) = \operatorname{tr}(BA)$", np.allclose(np.trace(_A @ _B), np.trace(_B @ _A)))}

    The cyclic identity looks like a curiosity here. It is not: it is the entire reason the
    trace survives a change of basis, since
    $\operatorname{{tr}}(P^{{-1}}AP) = \operatorname{{tr}}(APP^{{-1}}) = \operatorname{{tr}}A$.
    Notebook 5 cashes that in.

    Drag $t$ to $0$ and both sides go to $1$ — no time, no change in volume. Drag it
    negative to run the flow backwards, and the volume factor inverts.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - The **volume element** is the top-degree wedge: area $u\wedge v$ in the plane, signed
      volume $u\wedge v\wedge w$ (the scalar triple product) in space — multilinear and
      alternating.
    - The **determinant** is the factor by which an operator scales that element,
      $T(u)\wedge T(v)\wedge T(w) = (\det T)(u\wedge v\wedge w)$. Hence
      $\lvert\det\rvert$ = volume scaling, its sign = orientation, $\det = 0$ = collapse,
      and $\det(TS) = \det T\det S$ — all geometric, all confirmed against `numpy`.
    - The **trace** is the determinant's first-order shadow,
      $\operatorname{tr}T = \frac{d}{dt}\big\rvert_0\det(I+tT)$: equivalently the
      **divergence** of the flow $x\mapsto Tx$, with a basis-free wedge formula and the
      bridge $\det e^{tT} = e^{t\operatorname{tr}T}$.
    - Both are defined **without coordinates**. "Product of the eigenvalues" and "sum of the
      diagonal" are *consequences*, recovered once we finally choose a basis.
    - The shear parameter is the moral of the whole notebook: it visibly distorts every
      picture, and neither the determinant nor the trace notices. They measure volume and
      its rate of change — nothing else.

    **Next (notebook 5):** we at last introduce **basis vectors**, the **coordinates** of a
    vector, and the **matrix** of an operator — recovering all of the above as familiar
    formulas — and discover that the operators which are "a scaling combined with a pure
    rotation" form a copy of the **complex numbers**.
    """)
    return


if __name__ == "__main__":
    app.run()
