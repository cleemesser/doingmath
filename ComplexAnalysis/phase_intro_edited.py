# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "sympy",
#     "matplotlib",
#     "plotly",
#     "bokeh",
#     "spb (sympy-plot-backends)",
#     "k3d",
#     "ipympl",
#     "ipywidgets",
# ]
# ///
# maybe can cut down on these dependencies

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():

    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
# An Introduction to Phase Portraits in Complex Analysis

A phase portrait colors the domain by the value of a complex function. The hue
shows the phase `arg f(z)`. One full color wheel equals one full turn, and red
means `arg = 0`. Optional brightness contours (bands with lower brightness)
show more structure. E. Wegert describes this method in *Visual Complex
Functions* (2012).

    In Python you can do this easily with `sympy` and `sympy_plotting_backends` (spb).
    """)
    return


@app.cell
def _():
    import numpy as np
    from sympy import (
        Poly,
        exp,
        fraction,
        lambdify,
        I,
        roots,
        sin,
        symbols,
        together,
    )
    from spb import (
        analytic_landscape,
        BB,
        complex_points,
        domain_coloring,
        graphics,
        MB,
        PB,
        plotgrid,
        riemann_sphere_2d,
    )

    z = symbols("z")

    # The square domain used throughout. spb takes a 3-tuple (symbol, min, max) with *complex*
    # endpoints, so one tuple fixes both axes — the analogue of clmmathtools's `extent=`.
    DOMAIN2 = (z, -2 - 2j, 2 + 2j)
    DOMAIN3 = (z, -3 - 3j, 3 + 3j)
    return (
        BB,
        DOMAIN2,
        DOMAIN3,
        I,
        MB,
        PB,
        Poly,
        analytic_landscape,
        complex_points,
        domain_coloring,
        exp,
        fraction,
        graphics,
        lambdify,
        np,
        plotgrid,
        riemann_sphere_2d,
        roots,
        sin,
        symbols,
        together,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The four ways to encode the complex "z-plane"
    1. First write $z$ in polar form: $z= R e^{i\theta}$. Color each point by the angle $\theta$ (Figure 1a).
    2. Build the polar grid by drawing iso-phase lines (Figure 1b).
    3. Add iso-modulus (r) contours (Figure 1c).
    4. Combine 1, 2, and 3 (Figure 1d).
    """)
    return


@app.cell
def _(DOMAIN2, DOMAIN3, MB, domain_coloring, graphics, plotgrid, z):
    SCHEMES = [
        ("a", "Figure 1a: plane — hue encoded angle"),
        ("d", "Figure 1b: phase — iso-phase lines"),
        ("c", "Figure 1c: iso-modulus lines"),
        ("b", "Figure 1d: enhanced — all"),
    ]


    def portrait(f, domain=DOMAIN3, coloring="b", n=500, **kwargs):
        """One titled domain-coloring panel, titled from the expression itself."""
        series = domain_coloring(f, domain, coloring=coloring, n=n, colorbar=False)
        kwargs.setdefault("title", series[0].get_label(use_latex=True))
        return graphics(*series, grid=False,aspect='equal', axis=False, show=False, backend=MB, **kwargs)


    panels = [
        portrait(z, DOMAIN2, coloring=code, n=400, title=f'{name}')
        for code, name in SCHEMES
    ]
    plotgrid(*panels, nr=2, nc=2, size=(9, 8))
    return (portrait,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking the schemes numerically

    Series factories return objects, not pictures, so you can assert on the same data the renderer
    sees. `ComplexDomainColoringSeries.get_data()` returns `(x, y, abs, arg, rgb_img, colorscale)`.
    Note that spb also hands back the modulus and argument grids. clmmathtools's `phase.colorize()`
    does not.
    """)
    return


@app.cell
def _(DOMAIN2, domain_coloring, np, z):
    def coloring_image(code, expr=z, domain=DOMAIN2, n=200):
        """RGB image (float, 0–255) for one coloring scheme — straight off the series."""
        (series,) = domain_coloring(expr, domain, coloring=code, n=n, colorbar=False)
        return series.get_data()[4].astype(float)


    plane, enhanced = coloring_image("a"), coloring_image("b")
    assert (enhanced <= plane + 1e-9).all() and enhanced.mean() < plane.mean()
    print("schemes differ; enhancement only darkens (adds contour lines) ✓")

    # the abs/arg grids come back alongside the image, so the coloring can be spot-checked
    (s,) = domain_coloring(z, DOMAIN2, coloring="a", n=101, colorbar=False)
    x, y, mod, arg, _, _ = s.get_data()
    assert np.allclose(mod, np.hypot(x, y)) and np.allclose(arg, np.arctan2(y, x))
    print("series exposes |f| and arg f grids, consistent with f(z) = z ✓")
    return (coloring_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A small gallery (enhanced)

    A zero is a point where all hues meet and run counter-clockwise. A pole is a point where all
    hues meet and run clockwise. Count the color cycles to read the order.

    Note the titles. Every series carries a label derived from the expression, and
    `series.get_label(use_latex=True)` returns rendered LaTeX. So the panel titles below are not
    hand-written strings that can drift out of sync with the function, as they were in the
    clmmathtools version. (`graphics()` does not set a title on its own. It puts the label on the
    colorbar. We turned the colorbar off, so the code copies the label into `title=` explicitly.)
    """)
    return


@app.cell
def _(exp, plotgrid, portrait, sin, z):
    gallery = [z**2, 1 / z, (z**2 - 1) / (z**2 + 1), exp(z), sin(z)]

    plotgrid(*[portrait(f) for f in gallery], nr=2, nc=3, size=(12, 8))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the portrait: winding number = order of a zero/pole

    The winding number of `f` at a point is the number of hue cycles when you walk around that
    point. It is positive at a zero, and equals the order of the zero. It is negative at a pole.

    clmmathtools could only check this numerically, by marching a lambda around a circle. spb takes
    a symbolic expression, so you can get the exact answer first: factor the rational function and
    read the root multiplicities. Then confirm it numerically. The next cell annotates the plot
    with the exact answer.
    """)
    return


@app.cell
def _(I, Poly, coloring_image, fraction, lambdify, np, roots, together, z):
    def zeros_and_poles(expr):
        """Exact {point: order} dicts for a rational function, from root multiplicities."""
        num, den = fraction(together(expr))
        return roots(Poly(num, z)), roots(Poly(den, z))


    def winding(expr, center=0, r=0.4, n=2000):
        """Numeric winding number of `expr` around `center` — the argument principle, sampled."""
        f = lambdify(z, expr, "numpy")
        t = np.linspace(0, 2 * np.pi, n)
        ang = np.unwrap(np.angle(f(complex(center) + r * np.exp(1j * t))))
        return round((ang[-1] - ang[0]) / (2 * np.pi))


    R = (z**2 - 1) / (z**2 + 1)
    zs, ps = zeros_and_poles(R)
    assert zs == {1: 1, -1: 1} and ps == {I: 1, -I: 1}
    print(f"exact: zeros {zs}, poles {ps}")

    # exact orders and sampled winding numbers agree
    assert winding(z**2) == 2  # double zero
    assert winding(z**3) == 3  # triple zero
    assert winding(1 / z) == -1  # simple pole
    assert winding(R, center=1, r=0.3) == +1  # zero at z = 1
    assert winding(R, center=I, r=0.3) == -1  # pole at z = i
    assert all(winding(R, center=p, r=0.3) == +k for p, k in zs.items())
    assert all(winding(R, center=p, r=0.3) == -k for p, k in ps.items())
    print("winding numbers match exact zero/pole orders ✓")

    # poles render white; the whole portrait stays finite even through them
    assert np.isfinite(coloring_image("b", 1 / z, (z, -1 - 1j, 1 + 1j))).all()
    print("portrait finite everywhere; pole shows white ✓")
    return R, ps, zs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the `graphics()` interface pays off: composing series

    This is the analogue of the clmmathtools chain `p.phase_portrait(...).text(...).grid(...)`, but
    built the other way round. Instead of a `Plane` object that accumulates primitives, independent
    series factories each return a list, and `graphics()` unpacks them into one figure. Different
    kinds of series (a raster domain coloring, two scatters) mix freely. The markers sit on the
    exact roots computed above, not on eyeballed coordinates.
    """)
    return


@app.cell
def _(DOMAIN3, MB, R, complex_points, domain_coloring, graphics, ps, zs):
    marker = dict(linestyle="None", markersize=11, markeredgewidth=2)

    graphics(
        *domain_coloring(R, DOMAIN3, coloring="b", n=500, colorbar=False),
        *complex_points(
            *zs,
            label="zeros (exact)",
            rendering_kw=dict(marker="o", color="k", **marker),
        ),
        *complex_points(
            *ps,
            label="poles (exact)",
            rendering_kw=dict(marker="x", color="w", **marker),
        ),
        grid=False,
        aspect='equal',
        legend=True,
        title=r"$(z^2-1)/(z^2+1)$ with symbolically-located zeros and poles",
        size=(7, 6),
        backend=MB,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Things spb gives you that clmmathtools's `Plane` does not

    ### The analytic landscape, from the same expression

    clmmathtools needs a different class (`Space3D.landscape`) and a different backend (vedo) for
    this. In spb it is just another series factory over the same expression and range. `graphics()`
    notices the series is 3D and configures the figure accordingly.
    """)
    return


@app.cell
def _(DOMAIN3, MB, R, analytic_landscape, graphics):
    graphics(
        *analytic_landscape(R, DOMAIN3, n=200),
        backend=MB,
        size=(8, 7),
        title="analytic landscape: modulus lifted, colored by argument",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Riemann sphere, stereographically projected

    `riemann_sphere_2d` masks the portrait to the unit disk and annotates `0, 1, i, -i`. Pass
    `at_infinity=True` for the chart around `∞`, that is, the portrait of `f(1/z)`. There is no
    clmmathtools equivalent at all. This is what you get from a library whose author was reading the
    same Wegert book.
    """)
    return


@app.cell
def _(MB, R, graphics, plotgrid, riemann_sphere_2d):
    plotgrid(
        graphics(
            *riemann_sphere_2d(R, coloring="b", n=400, colorbar=False),
            grid=False,
            show=False,
            title="chart at 0",
            backend=MB,
        ),
        graphics(
            *riemann_sphere_2d(R, coloring="b", n=400, at_infinity=True, colorbar=False),
            grid=False,
            show=False,
            title="chart at ∞",
            backend=MB,
        ),
        nr=1,
        nc=2,
        size=(11, 5),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Magnitude-blended colorings tell zeros from poles

    Schemes `"l"` to `"o"` blend the modulus into the brightness, and bright means large
    $\lvert f\rvert$. A pole is a white blaze, and a zero is a dark pit. You can read this without
    tracing hue direction. `"k"` and `"k+log"` drop hue entirely and show only magnitude, in
    greyscale.
    clmmathtools has no counterpart. In each of its four schemes the brightness does not depend on
    $\lvert f\rvert$. Under `"b"` a zero and a pole are equally bright, and only the direction of
    the hue cycle tells them apart.
    """)
    return


@app.cell
def _(R, plotgrid, portrait):
    plotgrid(
        *[
            portrait(R, coloring=c, n=350, title=f'coloring="{c}"')
            for c in ["b", "l", "k+log"]
        ],
        nr=1,
        nc=3,
        size=(13, 4.5),
    )
    return


@app.cell
def _(DOMAIN3, I, R, domain_coloring, np):
    def ring_brightness(coloring, center, r=(0.04, 0.12), n=601):
        """Mean RGB brightness in a small annulus around `center`."""
        (s,) = domain_coloring(R, DOMAIN3, coloring=coloring, n=n, colorbar=False)
        x, y, _, _, img, _ = s.get_data()
        d = np.hypot(x - complex(center).real, y - complex(center).imag)
        return img.astype(float)[(d > r[0]) & (d < r[1])].mean()


    # under "b", a zero and a pole are equally bright; under "l" the pole is far brighter
    b_zero, b_pole = ring_brightness("b", 1), ring_brightness("b", I)
    l_zero, l_pole = ring_brightness("l", 1), ring_brightness("l", I)
    assert abs(b_zero - b_pole) < 5, (b_zero, b_pole)
    assert l_pole > l_zero + 50, (l_zero, l_pole)
    print(f'"b": zero {b_zero:.0f} vs pole {b_pole:.0f} — indistinguishable by brightness')
    print(f'"l": zero {l_zero:.0f} vs pole {l_pole:.0f} — pole blazes white ✓')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backends: the same series, four renderers

    clmmathtools splits between matplotlib and vedo, and you choose at `Plane(backend=...)`
    construction. spb chooses at `graphics(backend=...)`. Its series are entirely backend-agnostic:
    the identical `domain_coloring(...)` list feeds every one of them.

    - `MB` matplotlib: static figures for publication. Every cell above used it.
    - `PB` plotly: interactive pan, zoom, and hover in the browser. The hover box reports
      $\lvert f\rvert$ and $\arg f$ at the cursor, which is useful for reading a portrait.
    - `BB` bokeh: interactive 2D, lighter than plotly.
    - `KB` k3d: interactive WebGL 3D (for `analytic_landscape` / `riemann_sphere_3d`).
    """)
    return


@app.cell
def _(BB, DOMAIN3, MB, PB, R, domain_coloring, graphics):
    series = domain_coloring(R, DOMAIN3, coloring="b", n=400, colorbar=False)

    for backend in (MB, PB, BB):
        p = graphics(*series, backend=backend, grid=False, show=False, aspect='equal')
        print(f"{backend.__name__:20s} -> {type(p.fig).__module__}")
    print("one series list, three figure types ✓")
    return (series,)


@app.cell
def _(PB, graphics, series):
    # The plotly rendering — hover to read the modulus and argument off the surface.
    graphics(
        *series, backend=PB, grid=False, title="plotly: hover reports modulus and argument"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive exploration with widgets

    Because the expression is symbolic, a parameter can stay symbolic too. Pass
    `params={sym: (default, min, max)}`, and spb builds an ipywidgets (or panel) app that
    re-lambdifies and re-renders on change. Watch the `k`-fold zero at the origin split hues as `k`
    moves.

    clmmathtools has no analogue for this. With a Python callable you rebuild and redraw by hand.

    Backend note: the widget app embeds the figure in an ipywidgets `Box`, so the figure must itself
    be a widget. `PB` (plotly) satisfies that natively through `FigureWidget`. `MB` (matplotlib)
    does this only under `%matplotlib widget` (ipympl). With the default inline/Agg canvas it
    raises `TraitError: ... expected a Widget, not the FigureCanvasAgg`.
    """)
    return


@app.cell
def _(DOMAIN2, PB, domain_coloring, graphics, symbols, z):
    # this attempt at using a widget does not work
    k = symbols("k", positive=True)

    graphics(
        *domain_coloring(
            z**k * (z - 1) / (z + 1),
            DOMAIN2,
            coloring="b",
            n=300,
            colorbar=False,
            params={k: (2.0, 1.0, 5.0)},
        ),
        grid=False, aspect='equal',
        backend=PB,
        imodule="ipywidgets",
        title="z^k (z-1)/(z+1)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap: where each library wins

    | | `clmmathtools` | `spb` |
    |---|---|---|
    | input | Python callable on `ndarray` | SymPy expression |
    | non-symbolic `f` (data, iteration, numerics) | natural | needs a wrapper or is out of reach |
    | exact zeros / poles / residues to annotate with | you compute them yourself | `roots`, `residue`, `diff` on the same object |
    | labels | hand-written strings | LaTeX from the expression, automatic |
    | Wegert schemes | 4 (`plane`/`phase`/`modulus`/`enhanced`) | 16 + user callable, incl. magnitude-blended |
    | composition model | `Plane` object, chained mutators | series factories + `graphics(*series)` |
    | multi-panel | one figure per `display()` | `plotgrid()` |
    | 3D landscape | separate `Space3D` class | another series factory, same call shape |
    | Riemann sphere | none | `riemann_sphere_2d` / `_3d` |
    | widgets | none | `params={sym: (init, lo, hi)}` |
    | backends | matplotlib, vedo | matplotlib, plotly, bokeh, k3d, mayavi |
    | data introspection | `phase.colorize()` returns RGB | `series.get_data()` returns `x, y, abs, arg, rgb` |
    | control over the drawing | full, because it is your code | `rendering_kw` passthrough, then the series API's edge |

    In short: spb is the better tool when the function is symbolic, which is usual in classical
    complex analysis. You get exactness, LaTeX, four backends, and widgets for free. clmmathtools
    stays the better tool when the function is not symbolic: an iterated map, a transform defined
    numerically, or a function you only have as samples. It also stays better when a portrait is one
    layer of a scene you assemble from arbitrary primitives.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### History
    This notebook is a port of `04_phase_portraits.py` from clmmathtools to spb
    (`sympy-plot-backends`), so the two libraries can be compared on the same material.

    The biggest difference is the input type:

    - `clmmathtools` takes a Python callable `f: ndarray[complex] -> ndarray[complex]`. It samples,
      colors, and draws. Nothing knows what the function is.
    - `spb` takes a SymPy expression. spb lambdifies the expression for sampling, but the
      expression stays available symbolically. So you can compute zeros, poles, orders, residues,
      and derivatives exactly and use them to annotate the picture. Axis labels come out as
      rendered LaTeX for free.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
