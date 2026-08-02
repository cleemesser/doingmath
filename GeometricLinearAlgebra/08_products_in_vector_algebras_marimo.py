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

# Geometric Linear Algebra 8 -- products in vector algebras, as a *reactive* marimo notebook.
#
#   uv run marimo edit GeometricLinearAlgebra/08_products_in_vector_algebras_marimo.py
#
# NOTE ON SOURCE: unlike 01-07, the jupytext original of this notebook is a prose draft --
# its three code cells are empty or marked `TODO(human)`, and section 4 of its own outline
# (the geometric product) was never written. The prose is translated faithfully; the
# numerical checks marked "filled in here" below are NEW, written to match the style of the
# rest of the series (every notebook verifies its own claims). Section 4 is left as an
# explicit stub rather than invented. See the summary at the bottom for the full list.
#
# Draws with `mathviz`, so it runs either in the repo environment (the tested path) or via
# `marimo edit --sandbox` using the PEP 723 header above; see 01_..._marimo.py for details.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Linear Algebra 8 — products in vector algebras

    Notebooks 3 and 4 used $u \wedge v$ freely, justified by pictures: a signed area, an
    oriented plane element, a volume box. That was honest as *motivation* but loose as
    *definition* — we never said what kind of **object** $u\wedge v$ is, only what it
    measures. This notebook says exactly what it is, and every property we assumed becomes
    a consequence rather than an assertion.

    **Outline**

    1. The dot product / inner product, and the **dual space**
    2. The **tensor product** — where bilinearity becomes linearity
    3. The **wedge product** and the exterior algebra, defined properly
    4. The geometric product *(outlined in the source, not yet written)*

    /// warning | About this notebook
    The jupytext original of notebook 8 is a **prose draft**: its code cells are empty or
    marked `TODO(human)`. The prose here is a faithful translation. The numerical checks
    are new — written to match the rest of the series, and labelled where they appear.
    ///
    """)
    return


@app.cell
def _():
    import io
    from itertools import combinations
    from math import comb

    import marimo as mo
    import numpy as np
    import sympy as sp

    import mathviz as mv  # shared plane-viz library (see ../mathviz)
    from wigglystuff import TangleLatex

    return TangleLatex, combinations, comb, io, mo, mv, np, sp


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

    return BLUE, FAINT, GREEN, GREY, ORANGE, PURPLE, RED, check, png


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The dot product and dual spaces

    We keep in mind vectors in the plane $\mathbb{R}^2$ and in space $\mathbb{R}^3$, writing
    $u,v,w \in V$ for vectors and $a,b \in \mathbb{R}$ for scalars. On $\mathbb{R}^n$ the
    standard basis hands us a default inner product — a bilinear function taking two
    vectors to a real number, closely tied to the projection operator of notebook 2.

    The linear functions $f : V \to \mathbb{R}$ themselves form a vector space, the **dual
    space** $V^*$. Its elements obey the same vector-space axioms, and when $V$ is
    finite-dimensional, $\dim V^* = \dim V$. Elements of $V^*$ are called **covectors**,
    or *forms* / *1-forms*.

    Given an inner product there is a one-to-one mapping between $V$ and $V^*$:

    $$u^*(v) = \langle u, v\rangle,$$

    where $u^* \in V^*$ is the unique linear functional making the identity work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Resolving the tension in this section — *draft, for review*
    The source carries a `TODO` here observing that its two opening sentences say different
    things about what the inner product is **for**. They do, and the distinction matters
    enough to state plainly:

    **The dual space needs no inner product.** $V^*$ is just "the linear maps $V \to
    \mathbb{R}$", and the pairing $V^* \times V \to \mathbb{R}$, $(\phi, v) \mapsto
    \phi(v)$, is **canonical** — it is function application, available the moment $V$
    exists. Nothing is chosen.

    **The inner product is extra structure**, and what it buys is an *identification*
    $V \cong V^*$ sending $u \mapsto u^* = \langle u, -\rangle$. That isomorphism is not
    canonical in the way the pairing is: change the inner product and you change the
    identification, even though $V$ and $V^*$ are untouched.

    So the two roles are: **measuring** (lengths and angles — genuinely metric, genuinely
    extra) and **identifying** $V$ with $V^*$ (a convenience the metric happens to provide).
    Conflating them is why $V$ and $V^*$ look like the same space in $\mathbb{R}^n$ with the
    standard basis — the identification is the identity matrix there, so it becomes
    invisible. **Notebook 9 is entirely about pulling those two apart**, and shows that in a
    non-orthonormal basis the difference is unmissable: columns transform one way, rows the
    other.

    The trace in §2 is the sharpest illustration. It needs the canonical pairing and
    **no metric at all**.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The picture the source asked for

    The source's first code cell reads *"draw diagrams which illustrate the geometric
    interpretation of the dot product; also show the projection operator as defined by the
    dot product."* Here it is, filled in.

    A covector is best drawn **not** as an arrow but as its **level lines**: the parallel
    lines $u^*(v) = 0, 1, 2, \dots$ where it takes integer values. Reading off a covector
    means counting how many lines the vector $v$ pierces. That picture survives without a
    metric — which the arrow picture does not.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo):
    dualw = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"u = \begin{bmatrix} \tangle{ux} \\[2pt] \tangle{uy} \end{bmatrix}"
                r" \qquad "
                r"v = \begin{bmatrix} \tangle{vx} \\[2pt] \tangle{vy} \end{bmatrix}"
            ),
            parameters={
                "ux": {"value": 1.6, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "u, x",
                       "color": {"light": "#6f5cbd", "dark": "#b9a8ff"}},
                "uy": {"value": 0.8, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "u, y",
                       "color": {"light": "#6f5cbd", "dark": "#b9a8ff"}},
                "vx": {"value": 0.5, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "v, x",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
                "vy": {"value": 2.0, "min_value": -3, "max_value": 3, "step": 0.1,
                       "digits": 1, "label": "v, y",
                       "color": {"light": "#246bce", "dark": "#75a7ff"}},
            },
            editor="inline",
            theme="auto",
        )
    )
    dualw
    return (dualw,)


@app.cell(hide_code=True)
def _(BLUE, FAINT, GREEN, PURPLE, RED, check, dualw, mo, mv, np, png):
    u1 = np.array([dualw.values["ux"], dualw.values["uy"]])
    v1 = np.array([dualw.values["vx"], dualw.values["vy"]])
    _uu = float(u1 @ u1)
    _proj = (v1 @ u1) / _uu * u1 if _uu > 1e-12 else np.zeros(2)

    _pl = mv.Plane(extent=3.2)
    # Level lines of the covector u* = <u, -> : the sets where u*(v) = integer.
    if _uu > 1e-12:
        _dir = np.array([-u1[1], u1[0]]) / np.sqrt(_uu)
        for _lev in range(-6, 7):
            _base = u1 * (_lev / _uu)
            _pl.line(_base, _dir, color=FAINT)
    _pl.line([0, 0], u1 / max(np.sqrt(_uu), 1e-9), color=PURPLE)
    _pl.vector(u1, color=PURPLE, label="u")
    _pl.vector(v1, color=BLUE, label="v")
    _pl.vector(_proj, color=GREEN, label="P_u v")
    _pl.vector(v1 - _proj, origin=_proj, color=RED)

    mo.hstack(
        [
            png(_pl),
            mo.md(
                rf"""
    $$u^*(v) = \langle u, v\rangle = {float(u1 @ v1):+.3f}
      \qquad
      P_u v = \frac{{\langle v,u\rangle}}{{\langle u,u\rangle}}\,u
      = ({_proj[0]:+.3f},\; {_proj[1]:+.3f})$$

    {check(r"$u^*$ is **linear**: $u^*(av + bw) = a\,u^*(v) + b\,u^*(w)$", np.allclose(u1 @ (2.0 * v1 + 3.0 * np.array([1.0, -1.0])), 2.0 * (u1 @ v1) + 3.0 * (u1 @ np.array([1.0, -1.0]))))}
    {check(r"the projection is **idempotent**, $P_u(P_u v) = P_u v$", np.allclose((_proj @ u1) / _uu * u1, _proj))}
    {check(r"the rejection is perpendicular, $\langle v - P_u v,\, u\rangle = 0$", np.allclose((v1 - _proj) @ u1, 0.0))}
    {check(r"$v$ lies on a level line of $u^*$, i.e. $u^*(v)$ is an integer", np.isclose(float(u1 @ v1), round(float(u1 @ v1)), atol=1e-9))}

    The faint parallel lines are the level sets of $u^*$. Note what happens as you drag
    $u$ **longer**: the level lines crowd *closer together*, because a longer $u$ makes
    $u^*$ a "steeper" functional that counts up faster. A covector's natural picture is a
    stack of lines whose **spacing is inversely** proportional to its size — the opposite
    of an arrow, and the first hint that $V^*$ is not $V$.
    """
            ),
        ],
        widths=[1, 1],
        align="center",
        gap=1.2,
    )
    return u1, v1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The tensor product: where bilinearity becomes linearity

    The dot product takes *two* vectors and returns a number, and it is **bilinear** —
    linear in each slot separately. So is the wedge product. So is $(T,v)\mapsto Tv$.
    Bilinear maps are everywhere, and they are **not** linear maps, so none of our machinery
    applies to them directly.

    The tensor product fixes this once and for all. Rather than study each bilinear map on
    its own terms, we build a *single new vector space* $V \otimes W$ in which the
    bilinearity has already been absorbed, so that every bilinear map on $V\times W$ becomes
    an ordinary **linear** map on $V \otimes W$.

    For each pair we write a formal product $v \otimes w$, and declare the only rules to be
    the ones bilinearity forces:

    $$(a v_1 + b v_2) \otimes w = a\,(v_1 \otimes w) + b\,(v_2 \otimes w), \qquad
      v \otimes (a w_1 + b w_2) = a\,(v \otimes w_1) + b\,(v \otimes w_2).$$

    $V \otimes W$ is all linear combinations of such symbols, **subject to nothing else**.
    That "nothing else" is the whole content, and it is what the **universal property**
    records: for every bilinear $B : V\times W \to U$ there is exactly one linear
    $\tilde B : V \otimes W \to U$ with $B(v,w) = \tilde B(v\otimes w)$. The tensor product
    forgets nothing about $B$ and adds nothing to it.

    Two consequences worth pinning down:

    - **Dimension multiplies.** $\{b_i \otimes c_j\}$ is a basis, so
      $\dim(V\otimes W) = \dim V \cdot \dim W$. (Compare $V \oplus W$, where dimensions
      *add*.)
    - **Most tensors are not simple.** An element of the form $v\otimes w$ is *simple* or
      *rank-one*, but these do not fill the space — they are not even a subspace. With
      $\dim V = \dim W = 3$ the simple tensors sweep out only a $5$-dimensional cone inside
      a $9$-dimensional space. The general element is a **sum** of simple tensors, and the
      fewest terms needed is its **rank**.

    Note $v \otimes w \neq w \otimes v$: the tensor product is not commutative, because we
    imposed no rule saying it should be. That freedom is what makes it the right raw
    material. The dot product is *symmetric*, the wedge *antisymmetric*; both come from
    imposing one extra rule on $V\otimes V$:

    $$V \otimes V \;=\; \operatorname{Sym}^2 V \;\oplus\; \Lambda^2 V, \qquad
      n^2 = \tfrac{n(n+1)}{2} + \tfrac{n(n-1)}{2}.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear transformations are $V \otimes V^*$

    Take $u \in V$ and a covector $\phi \in V^*$, and build

    $$(u \otimes \phi)(v) \;=\; \phi(v)\, u .$$

    Read it left to right: $\phi$ **consumes** the input and returns a number; that number
    scales $u$, which is **produced** as the output. In coordinates this is the *outer
    product* $u\,\phi^{\mathsf{T}}$ — a column times a row, a matrix of rank one.

    Every linear map arises this way. With $\{b_i\}$ a basis and $\{\beta^j\}$ the dual
    basis ($\beta^j(b_i) = \delta^j_i$), the maps $b_i \otimes \beta^j$ are precisely the
    matrix units $E_{ij}$, and any $T$ expands as
    $T = \sum_{i,j} T^i{}_j\, b_i \otimes \beta^j$.

    So the space of linear transformations of $V$ **is** $V \otimes V^*$, and the dimension
    agrees: $n \cdot n = n^2$. This is also why index notation puts one index up and one
    down — the slots are not interchangeable. It is $V \otimes V^*$ and **not**
    $V \otimes V$: one factor eats a vector, the other emits one.

    Two payoffs fall out immediately.

    **The projection operator of notebook 2 was a rank-one tensor all along:**

    $$P_u \;=\; \frac{u \otimes u^*}{\langle u, u\rangle}.$$

    **The trace is the canonical contraction.** There is one obvious thing to do with an
    element of $V \otimes V^*$ — feed the $V$ part to the $V^*$ part:

    $$\operatorname{tr}(u \otimes \phi) \;=\; \phi(u),$$

    extended linearly. On the identity $I = \sum_i b_i \otimes \beta^i$ this gives
    $\sum_i \beta^i(b_i) = n$, as it must.

    This is stronger than it looks. Notebook 4 defined the trace coordinate-free *through
    volume*. Here we get it with no volume form, no basis, and — crucially — **no inner
    product**: the pairing of $V$ against $V^*$ is built into the space. Contrast
    $V \otimes V$, which has no canonical pairing at all; you would have to choose a metric
    first. **The trace is free; the "trace" of a $(2,0)$ tensor is not.**
    """)
    return


@app.cell
def _(check, comb, mo, np, u1):
    # ── filled in here: the source's first TODO(human) numerical check ────────────
    _rng = np.random.default_rng(0)
    n = 3
    u, phi = _rng.normal(size=n), _rng.normal(size=n)
    v = _rng.normal(size=n)

    outer = np.outer(u, phi)  # the tensor u (x) phi, in coordinates

    # A general T, expanded in the basis b_i (x) beta^j of matrix units.
    T = _rng.normal(size=(n, n))
    rebuilt = sum(
        T[i, j] * np.outer(np.eye(n)[i], np.eye(n)[j]) for i in range(n) for j in range(n)
    )

    # The projection of notebook 2, rebuilt as a rank-one tensor.
    P_tensor = np.outer(u, u) / (u @ u)
    P_nb2 = np.array([(w @ u) / (u @ u) * u for w in np.eye(n)]).T

    mo.md(
        rf"""
    **The tensor picture and the matrix picture agree**

    {check(r"$(u\otimes\phi)(v) = \phi(v)\,u$ — the definition equals the outer product acting on $v$", np.allclose(outer @ v, (phi @ v) * u))}
    {check(r"$u\otimes\phi$ has **rank one**, as a simple tensor must", np.linalg.matrix_rank(outer) == 1)}
    {check(r"$u\otimes\phi \neq \phi\otimes u$ — the tensor product is not commutative", not np.allclose(outer, np.outer(phi, u)))}
    {check(r"$T = \sum_{i,j} T^i{}_j\, b_i\otimes\beta^j$ rebuilds any linear map", np.allclose(rebuilt, T))}
    {check(r"$\operatorname{tr}(u\otimes\phi) = \phi(u)$ — the canonical contraction", np.allclose(np.trace(outer), phi @ u))}
    {check(rf"$\operatorname{{tr}} I = n = {n}$", np.isclose(np.trace(np.eye(n)), n))}
    {check(r"$P_u = \dfrac{u\otimes u^*}{\langle u,u\rangle}$ **is** the projection of notebook 2", np.allclose(P_tensor, P_nb2))}
    {check(r"and it is idempotent, $P_u^2 = P_u$", np.allclose(P_tensor @ P_tensor, P_tensor))}

    **Dimension counts**

    {check(rf"$\dim(V\otimes V) = n^2 = {n**2}$, while $\dim(V\oplus V) = 2n = {2 * n}$ — tensor multiplies, direct sum adds", n**2 == 9 and 2 * n == 6)}
    {check(rf"$n^2 = \tfrac{{n(n+1)}}{{2}} + \tfrac{{n(n-1)}}{{2}}$, i.e. ${n**2} = {n * (n + 1) // 2} + {n * (n - 1) // 2}$ — $\operatorname{{Sym}}^2 \oplus \Lambda^2$", n**2 == n * (n + 1) // 2 + n * (n - 1) // 2)}
    {check(rf"the simple tensors form a **{2 * n - 1}-dimensional** cone inside {n**2} dimensions — most tensors are not simple", 2 * n - 1 == 5 and n**2 == 9)}

    That last count is the one to sit with. A rank-one $3\times3$ matrix $u\,\phi^{{\mathsf{{T}}}}$
    is determined by $3 + 3 = 6$ numbers, minus $1$ for the scaling you can shuffle between
    $u$ and $\phi$: **{2 * n - 1} dimensions** inside a **{n**2}-dimensional** space. The
    simple tensors are a thin, curved cone, not a subspace — add two of them and you
    generically leave it. That is exactly why "rank" is a meaningful and non-trivial
    invariant.
    """
    )
    return T, n, outer, phi, u, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The wedge product and the exterior algebra, defined properly

    Start from the **tensor algebra**, the direct sum of all tensor powers of $V$,

    $$T(V) \;=\; \bigoplus_{k \ge 0} V^{\otimes k}
      \;=\; \mathbb{R} \oplus V \oplus (V\otimes V) \oplus \cdots,$$

    with $\otimes$ itself as multiplication. This is the free-est possible algebra built on
    $V$ — it imposes no relations at all, which is why §2 could call it "raw material."

    The exterior algebra is what you get by imposing **one** relation:

    $$\Lambda V \;=\; T(V) \,\big/\, \big\langle\, v \otimes v \;:\; v \in V \,\big\rangle$$

    — the quotient by the two-sided ideal generated by all squares. We write the induced
    product as $\wedge$. That single rule, $v\wedge v = 0$, generates everything.

    ### Antisymmetry is a theorem, not an axiom

    This derivation is worth doing explicitly, because the two properties are usually stated
    as if interchangeable. Apply the rule to $u+v$:

    $$0 = (u+v)\wedge(u+v) = \underbrace{u\wedge u}_{0} + u\wedge v + v\wedge u
      + \underbrace{v\wedge v}_{0} \quad\Longrightarrow\quad u\wedge v = -\,v\wedge u.$$

    So **alternating implies antisymmetric**. The converse needs division by $2$, so over
    $\mathbb{R}$ they are equivalent — but $v\wedge v = 0$ is the stronger, more primitive
    statement, and it says directly what the geometry demanded: a parallelogram with two
    equal edges has no area.

    ### As a subspace instead of a quotient

    Over $\mathbb{R}$ we may equally realise $\Lambda^2 V$ *inside* $V \otimes V$ as the
    antisymmetric tensors, via

    $$u \wedge v \;=\; u \otimes v - v \otimes u .$$

    This is a genuine **convention**: many texts insert $\tfrac12$ (or $\tfrac{1}{k!}$ in
    degree $k$) so the identification with the quotient is an isometry rather than merely an
    isomorphism. Nothing conceptual hangs on it, but determinant and norm formulas pick up
    powers of $k!$ depending on the choice — one of the places loose definitions actually
    bite. We use the factor-free convention.

    This is also where §2's warning about simple tensors returns. $V \otimes V$ splits as
    $\operatorname{Sym}^2 V \oplus \Lambda^2 V$; the wedge is exactly the projection onto
    the second summand, and the dot product — symmetric — lives in the first.
    **The two products of this series are the two halves of the tensor product.**
    """)
    return


@app.cell
def _(check, mo, np):
    # ── filled in here: part of the source's second TODO(human) ───────────────────
    _rng = np.random.default_rng(1)
    a, b = _rng.normal(size=3), _rng.normal(size=3)

    def wedge(x, y):
        """Lambda^2 V realised inside V (x) V as the antisymmetric tensors."""
        return np.outer(x, y) - np.outer(y, x)

    def sym(x, y):
        return np.outer(x, y) + np.outer(y, x)

    _W = wedge(a, b)
    _S = sym(a, b)

    mo.md(
        rf"""
    **$v\wedge v = 0$ generates antisymmetry**

    {check(r"$v\wedge v = 0$ — the single axiom", np.allclose(wedge(a, a), 0))}
    {check(r"$(u+v)\wedge(u+v) = 0$ too", np.allclose(wedge(a + b, a + b), 0))}
    {check(r"**therefore** $u\wedge v = -\,v\wedge u$ — antisymmetry, derived", np.allclose(_W, -wedge(b, a)))}
    {check(r"the wedge really is antisymmetric as a matrix, $W^{\mathsf{T}} = -W$", np.allclose(_W.T, -_W))}
    {check(r"and the symmetric part is genuinely symmetric, $S^{\mathsf{T}} = S$", np.allclose(_S.T, _S))}
    {check(r"$u\otimes v = \tfrac12(S + W)$ — the tensor splits into its two halves", np.allclose(np.outer(a, b), 0.5 * (_S + _W)))}
    {check(r"the two halves are independent: $\langle S, W\rangle = 0$ under the trace inner product", np.isclose(np.sum(_S * _W), 0.0))}

    The last two ticks are the decomposition $V\otimes V = \operatorname{{Sym}}^2 V \oplus
    \Lambda^2 V$ made concrete. Any tensor is the sum of a symmetric and an antisymmetric
    part, the split is unique, and the two summands are orthogonal. The dot product lives
    entirely in the first; the wedge entirely in the second.
    """
    )
    return (wedge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Higher degrees and the shape of $\Lambda V$

    The same quotient in degree $k$ kills any product with a repeated factor, leaving
    $\dim \Lambda^k V = \binom{n}{k}$, with basis
    $\{b_{i_1}\wedge\cdots\wedge b_{i_k}\}$ over strictly increasing index tuples. Summing
    over $k$ gives $\dim \Lambda V = 2^n$: the exterior algebra of an $n$-dimensional space
    has **one basis element per subset** of a basis of $V$.

    The product is associative and **graded-commutative**: for $\alpha \in \Lambda^k$ and
    $\beta \in \Lambda^l$,

    $$\alpha \wedge \beta = (-1)^{kl}\, \beta \wedge \alpha.$$

    Vectors anticommute; bivectors commute with each other.

    Three things this immediately explains, all previously taken on faith:

    - **The determinant.** $\Lambda^n V$ is one-dimensional, so any linear $T$ acts on it by
      a single number. That number is $\det T$ — notebook 4's volume definition, recovered
      as a **dimension count**.
    - **Why the cross product is a three-dimensional accident.**
      $\dim \Lambda^2\mathbb{R}^3 = 3 = \dim\mathbb{R}^3$, so in 3D — and, checking
      $\binom{n}{2} = n$, in *no other dimension* — a bivector can be disguised as a vector.
      That disguise is the cross product, and it is why $u\times v$ fails to generalise
      while $u\wedge v$ does.
    - **Bivectors are not all planes.** In $\mathbb{R}^3$ every bivector is simple, so
      "an oriented plane patch" is safe there. It fails at $n \ge 4$: in $\mathbb{R}^4$ the
      bivector $e_1\wedge e_2 + e_3\wedge e_4$ is not simple and represents no plane at all.
    """)
    return


@app.cell
def _(check, comb, mo, np):
    # ── filled in here: the rest of the source's second TODO(human) ───────────────
    def dims(n):
        return [comb(n, k) for k in range(n + 1)]

    _rows = []
    for _n in range(1, 7):
        _d = dims(_n)
        _rows.append(
            f"| {_n} | {' , '.join(str(x) for x in _d)} | {sum(_d)} | "
            f"{'**yes**' if comb(_n, 2) == _n else 'no'} |"
        )

    # det as the action on the 1-dimensional top degree, via the triple product.
    _rng = np.random.default_rng(2)
    _T = _rng.normal(size=(3, 3))
    _e = np.eye(3)
    _top = np.dot(_T @ _e[0], np.cross(_T @ _e[1], _T @ _e[2]))
    _base = np.dot(_e[0], np.cross(_e[1], _e[2]))

    # Simplicity of a 2-form in R^4: B is simple  <=>  B ^ B = 0  <=>  Pfaffian = 0.
    def pfaffian4(B):
        return B[0, 1] * B[2, 3] - B[0, 2] * B[1, 3] + B[0, 3] * B[1, 2]

    _B_nonsimple = np.zeros((4, 4))
    _B_nonsimple[0, 1], _B_nonsimple[1, 0] = 1, -1  # e1 ^ e2
    _B_nonsimple[2, 3], _B_nonsimple[3, 2] = 1, -1  # e3 ^ e4

    _x, _y = _rng.normal(size=4), _rng.normal(size=4)
    _B_simple = np.outer(_x, _y) - np.outer(_y, _x)  # a genuine u ^ v

    mo.md(
        rf"""
    | $n$ | $\dim\Lambda^k V$ for $k = 0\ldots n$ | $\dim\Lambda V$ | is $\binom{{n}}{{2}} = n$? |
    |---|---|---|---|
    {chr(10).join(_rows)}

    {check(r"$\sum_k \binom{n}{k} = 2^n$ — one basis element per subset", all(sum(dims(_n)) == 2**_n for _n in range(1, 9)))}
    {check(r"$\dim\Lambda^n V = 1$ — the top degree is a line, so $T$ acts on it by one number", all(comb(_n, _n) == 1 for _n in range(1, 9)))}
    {check(rf"that number is $\det T$: the triple product scales by ${_top / _base:+.4f}$, and $\det T = {np.linalg.det(_T):+.4f}$", np.allclose(_top / _base, np.linalg.det(_T)))}
    {check(r"$\binom{n}{2} = n$ **only** at $n = 3$ — the cross product's accident", [_n for _n in range(1, 40) if comb(_n, 2) == _n] == [3])}
    {check(r"in $\mathbb{R}^4$, $e_1\wedge e_2 + e_3\wedge e_4$ is **not simple** (its Pfaffian is nonzero)", not np.isclose(pfaffian4(_B_nonsimple), 0.0))}
    {check(r"whereas a genuine $u\wedge v$ in $\mathbb{R}^4$ **is** simple (Pfaffian zero)", np.isclose(pfaffian4(_B_simple), 0.0))}

    The $\binom{{n}}{{2}} = n$ tick is the whole story of the cross product in one line. Solving
    $\tfrac{{n(n-1)}}{{2}} = n$ gives $n = 3$ (or the degenerate $n=0$), and that **single
    coincidence** is the reason three-dimensional vector calculus has a cross product at all.
    It is not a deep fact about space; it is a solution to a small quadratic. The wedge, which
    never needed the coincidence, generalises to every dimension.

    The Pfaffian ticks make "not simple" concrete: a bivector $B$ is a single plane element
    exactly when $B\wedge B = 0$, and in $\mathbb{{R}}^4$ that is the Pfaffian
    $B_{{01}}B_{{23}} - B_{{02}}B_{{13}} + B_{{03}}B_{{12}}$. For $e_1\wedge e_2 + e_3\wedge e_4$
    it equals ${pfaffian4(_B_nonsimple):.0f}$ — so that bivector is a genuine sum of two plane
    elements lying in **completely disjoint** planes, and no single parallelogram anywhere in
    $\mathbb{{R}}^4$ represents it.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The geometric product — *not yet written*

    The source's outline lists a fourth section on the **geometric product**, but the draft
    stops after section 3. It is left as a stub here rather than invented, so that this
    notebook stays a translation rather than an extension.

    For orientation, the missing content is the observation notebook 3 already gestured at:

    $$u\,v \;=\; \langle u,v\rangle \;+\; u\wedge v,$$

    a scalar plus a bivector, added together in one algebra — the move that makes the
    product **invertible** and turns rotations into conjugations $v \mapsto RvR^{-1}$. The
    `Geometry/` directory of this repo works with exactly this structure via
    [`kingdon`](https://github.com/tBuLi/kingdon), if you want the computational version
    before the expository one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - **$V^*$ needs no metric.** The pairing $\phi(v)$ is canonical; the inner product is
      *extra* structure whose job is to **identify** $V$ with $V^*$. Those two roles get
      conflated in $\mathbb{R}^n$ with the standard basis because the identification is the
      identity matrix there. Notebook 9 pulls them apart.
    - **The tensor product** absorbs bilinearity into a new space, so every bilinear map
      becomes an ordinary linear one. Dimensions **multiply**, and **most tensors are not
      simple** — the rank-one cone is $2n-1$ dimensional inside $n^2$.
    - **Linear maps are $V\otimes V^*$**, one slot consuming and one producing. Projection is
      a rank-one tensor $u\otimes u^*/\langle u,u\rangle$, and the **trace is the canonical
      contraction** — available with no basis, no volume form and **no inner product**,
      which is strictly stronger than notebook 4's volume definition.
    - **The exterior algebra is one quotient**: $\Lambda V = T(V)/\langle v\otimes v\rangle$.
      From the single axiom $v\wedge v = 0$, antisymmetry is a **theorem**. $\dim\Lambda^k V
      = \binom{n}{k}$, $\dim\Lambda V = 2^n$, and $\Lambda^n V$ being a line is *why* the
      determinant is a single number.
    - **The cross product is an accident of $\binom{n}{2} = n$**, which holds only at
      $n = 3$; and from $n \ge 4$ bivectors need not be plane elements at all.

    ### What was added versus translated

    | part | status |
    |---|---|
    | all prose in §§1–3 | translated from the jupytext source |
    | §1 resolution of the source's prose `TODO` | **drafted here**, for your review |
    | §1 dot-product / projection picture | **new** (the source cell was a comment only) |
    | §2 tensor-vs-matrix numerical checks | **new** (the source cell was `TODO(human)`) |
    | §3 antisymmetry, dimension and Pfaffian checks | **new** (the source cell was `TODO(human)`) |
    | §4 geometric product | **left as a stub** — never written in the source |

    **Next (notebook 9):** the concrete payoff of §1 — a basis for $V$ drags along a **dual
    basis** for $V^*$, columns transform one way and rows the other, and the pairing that
    survives both is row-times-column, with no metric anywhere in sight.
    """)
    return


if __name__ == "__main__":
    app.run()
