# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "sympy",
#     "matplotlib",
#     "plotly",
#     "bokeh",
#     "sympy_plot_backends", # spb package
#     "k3d",
#     "ipympl",
#     "ipywidgets",
# ]
# ///
# maybe can cut down on these dependencies

import marimo


__generated_with = "0.24.0"
app = marimo.App(app_title="Intro to Phase Portraits", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo


    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # An Introduction to Phase Portraits in Complex Analysis

    A **phase portrait** colors the domain by the value of a complex function: the
    **hue** is the phase `arg f(z)` (a full color wheel per turn, red at `arg = 0`),
    and optional brightness **contours** encode more structure. Following E. Wegert,
    *Visual Complex Functions* (2012).

        In python, this is really easy to do using `sympy` and `sympy_plotting_backends` (spb).

    In the code below we start by importing sympy objects we want, activate fancy printing and then define the symbol `z` to represent points in the complex plane:
    ```python
    z = symbols("z")
    ```
    This is how sympy indicates creates an object in python which represents a mathematical symbol.

    We then define two domains `DOMAIN2` and `DOMAIN3` which are square areas of the complex plane.
    """)
    return


@app.cell
def _():
    import numpy as np

    # in marimo can't do a from sympy import *
    import sympy as sp
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

    sp.init_printing()

    # The `spb` module is shipped by the distribution `sympy_plot_backends`,
    # which the PEP 723 header above declares. PEP 723 has no way to record
    # that an import name differs from a distribution name, so marimo's
    # package manager guesses a PyPI name from the import name: on molab a
    # plain `from spb import ...` makes it install the unrelated project
    # literally named `spb`, and the import still fails. Going through
    # importlib hides the import from that static scan, and the header
    # installs the right package.
    import importlib

    spb = importlib.import_module("spb")

    analytic_landscape = spb.analytic_landscape
    BB = spb.BB
    complex_points = spb.complex_points
    domain_coloring = spb.domain_coloring
    graphics = spb.graphics
    MB = spb.MB
    PB = spb.PB
    plotgrid = spb.plotgrid
    riemann_sphere_2d = spb.riemann_sphere_2d

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
    ### Overview
    1. We consider $z$ in its polar form first, $z= R e^{i\theta}$. Then we encode each point with a color based upon the angle $\theta$ (figure 1a)
    2. Start building the polar grid by drawing iso-phase lines (Figure 1b)
    3. Put in iso-modulus (r) contours (Figure 1c)
    4. Finally we put these all together in a standard phase portrait: 1+2+3 (Figure 1d)

    ### Encoding the polar angle with a hue
    Since we can write a number in the complex plane as $z = Re^{i\theta}$, we can assign a hue from a given color scheme to every value of $\theta$ from 0 to $2\pi$. `spb` does this in the code below:
    ```python
    graphics(
        *domain_coloring(z, # the function, z, to use for coloring
                         (z, -2-2j, 2+2j), # DOMAIN2 written out
                         coloring='a',
                         n=500,
                         colorbar=False),
        title="Figure 1a: plane - hue encoded angle",
        aspect="equal",
        grid=False
    )
    ```
    """)
    return


@app.cell
def _(DOMAIN2, domain_coloring, graphics, z):
    graphics(
        *domain_coloring(z, DOMAIN2, coloring="a", n=500, colorbar=False),
        title="Figure 1a: plane - hue encoded angle",
        aspect="equal",
        grid=False,
        show=False,  # don't need to show because of last line
    ).fig  # need to make it so the figure is evaluated as the value of the cell for it to appear in "app" mode
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can start creating a "polar grid" to the complex plane by adding "iso-phase" lines. Lines where $\theta$ is constant. I am also going to remove the tick marks, leaving the domain implicit.
    """)
    return


@app.cell
def _(DOMAIN2, domain_coloring, graphics, z):
    graphics(
        *domain_coloring(z, DOMAIN2, coloring="d", n=500, colorbar=False),
        title="Figure 1b: plane - hue encoded angle + iso-phase lines",
        aspect="equal",
        axis=False,  # turn off the axis
        grid=False,
        show=False,  # don't need to show twice
    ).fig
    return


@app.cell
def _(DOMAIN2, domain_coloring, graphics, z):
    graphics(
        *domain_coloring(z, DOMAIN2, coloring="c", n=500, colorbar=False),
        title="Figure 1c: plane - hue encoded angle + iso-magnitude lines",
        aspect="equal",
        axis=False,  # turn off the axis
        grid=False,
        show=False,
    ).fig
    return


@app.cell
def _(DOMAIN2, domain_coloring, graphics, z):
    graphics(
        *domain_coloring(z, DOMAIN2, coloring="b", n=500, colorbar=False),
        title="Figure 1d: plane - hue-angle + polar grid",
        aspect="equal",
        axis=False,  # turn off the axis
        grid=False,
        show=False,  # don't show twice
    ).fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking the schemes numerically

    Because series factories return objects rather than pictures, the *same* data the renderer sees is
    available for assertions. `ComplexDomainColoringSeries.get_data()` returns
    `(x, y, abs, arg, rgb_img, colorscale)` — note that spb hands back the modulus and argument grids
    too.
    """)
    return


@app.cell
def _(DOMAIN3, domain_coloring, graphics, mo, z):
    # how to enable matplotlib interactive pan/zoom/save
    mo.mpl.interactive(
        graphics(
            *domain_coloring(z**2, DOMAIN3, coloring="b", n=2500, colorbar=False),
            grid=False, axis=False, aspect="equal", show=False,
        ).fig
    )
    return


@app.cell
def _(DOMAIN2, DOMAIN3, MB, domain_coloring, graphics, z):
    def portrait(f, domain=DOMAIN3, coloring="b", n=500, **kwargs):
        """One titled domain-coloring panel, titled from the expression itself."""
        series = domain_coloring(f, domain, coloring=coloring, n=n, colorbar=False)
        kwargs.setdefault("title", series[0].get_label(use_latex=True))
        return graphics(
            *series,
            grid=False,
            aspect="equal",
            axis=False,
            show=False,
            backend=MB,
            **kwargs,
        )


    def coloring_image(code, expr=z, domain=DOMAIN2, n=200):
        """RGB image (float, 0-255) for one coloring scheme - straight off the series."""
        (series,) = domain_coloring(expr, domain, coloring=code, n=n, colorbar=False)
        return series.get_data()[4].astype(float)

    return coloring_image, portrait


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A small gallery (enhanced)

    Zeros are where all hues meet running **counter-clockwise**; poles are where they meet running
    **clockwise**. Count the color cycles to read the order.

    Note the titles. Every series carries a label derived from the expression, and
    `series.get_label(use_latex=True)` hands back rendered LaTeX — so the panel titles below are not
    hand-written strings that can drift out of sync with the function, as they were in the clmmathtools
    version. (`graphics()` does not set a title on its own; it puts the label on the colorbar, which
    we have turned off, so we lift it into `title=` explicitly.)
    """)
    return


@app.cell
def _(exp, plotgrid, portrait, sin, z):
    gallery = [z**2, 1 / z, (z**2 - 1) / (z**2 + 1), exp(z), sin(z)]

    plotgrid(*[portrait(f) for f in gallery], nr=3, nc=2, size=(8, 12))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the portrait: winding number = order of a zero/pole

    The number of times the hue cycles as you circle a point is the **winding number** of `f` there —
    positive for a zero of that order, negative for a pole.

    clmmathtools could only check this **numerically**, by marching a lambda around a circle. With a
    symbolic expression in hand, spb's input lets us get the answer **exactly** first — factor the
    rational function and read off root multiplicities — and then confirm it numerically. The exact
    answer is what we annotate the plot with in the next cell.
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

    This is the analogue of clmmathtools's `p.phase_portrait(...).text(...).grid(...)` chain, but built the
    other way round: instead of a `Plane` object accumulating primitives, independent series factories
    each return a list, and `graphics()` splats them into one figure. Different *kinds* of series
    (a raster domain coloring, two scatters) mix freely, and the markers are placed from the **exact**
    roots computed above rather than eyeballed coordinates.
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
        aspect="equal",
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

    clmmathtools needs a *different class* (`Space3D.landscape`) and a different backend (vedo) for this.
    In spb it is just another series factory over the same expression and range — `graphics()` notices
    the series is 3D and configures the figure accordingly.
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

    `riemann_sphere_2d` masks the portrait to the unit disk and annotates `0, 1, i, -i`; pass
    `at_infinity=True` for the chart around `∞` (i.e. the portrait of `f(1/z)`). There is no clmmathtools
    equivalent at all — this is the payoff of a library whose author was reading the same Wegert book.
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
            *riemann_sphere_2d(
                R, coloring="b", n=400, at_infinity=True, colorbar=False
            ),
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

    Schemes `"l"`–`"o"` blend the modulus into the brightness: bright means large $\lvert f\rvert$.
    So a pole is a white blaze and a zero a dark pit, readable without tracing hue direction.
    `"k"` / `"k+log"` drop hue entirely and show magnitude alone in greyscale.
    clmmathtools has no counterpart; its four schemes all keep brightness independent of $\lvert f\rvert$
    — under `"b"` a zero and a pole are equally bright, and only the *direction* of the hue cycle
    distinguishes them.
    """)
    return


@app.cell
def _(R, plotgrid, portrait):
    plotgrid(
        *[
            portrait(R, coloring=c, n=350, title=f'coloring="{c}"')
            for c in ["b", "l", "k+log"]
        ],
        nr=3,
        nc=1,
        size=(4.5, 13),
    )
    return


@app.cell
def _(DOMAIN3, I, R, domain_coloring, np):
    def ring_brightness(coloring, center, r=(0.04, 0.12), n=601):
        """Mean RGB brightness in a small annulus around `center`."""
        (s,) = domain_coloring(
            R, DOMAIN3, coloring=coloring, n=n, colorbar=False
        )
        x, y, _, _, img, _ = s.get_data()
        d = np.hypot(x - complex(center).real, y - complex(center).imag)
        return img.astype(float)[(d > r[0]) & (d < r[1])].mean()

    # under "b", a zero and a pole are equally bright; under "l" the pole is far brighter
    b_zero, b_pole = ring_brightness("b", 1), ring_brightness("b", I)
    l_zero, l_pole = ring_brightness("l", 1), ring_brightness("l", I)
    assert abs(b_zero - b_pole) < 5, (b_zero, b_pole)
    assert l_pole > l_zero + 50, (l_zero, l_pole)
    print(
        f'"b": zero {b_zero:.0f} vs pole {b_pole:.0f} — indistinguishable by brightness'
    )
    print(f'"l": zero {l_zero:.0f} vs pole {l_pole:.0f} — pole blazes white ✓')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backends: the same series, four renderers

    clmmathtools's backend split is matplotlib vs vedo, chosen at `Plane(backend=...)` construction. spb's
    is chosen at `graphics(backend=...)`, and the series are entirely backend-agnostic — the identical
    `domain_coloring(...)` list feeds every one of them.

    - **`MB`** matplotlib — static, publication figures (what every cell above used)
    - **`PB`** plotly — interactive pan/zoom/hover in the browser; the hover box reports
      $\lvert f\rvert$ and $\arg f$ at the cursor, which is genuinely useful for reading a portrait
    - **`BB`** bokeh — interactive 2D, lighter-weight than plotly
    - **`KB`** k3d — interactive WebGL 3D (for `analytic_landscape` / `riemann_sphere_3d`)
    """)
    return


@app.cell
def _(BB, DOMAIN3, MB, PB, R, domain_coloring, graphics):
    series = domain_coloring(R, DOMAIN3, coloring="b", n=400, colorbar=False)

    for backend in (MB, PB, BB):
        p = graphics(
            *series, backend=backend, grid=False, show=False, aspect="equal"
        )
        print(f"{backend.__name__:20s} -> {type(p.fig).__module__}")
    print("one series list, three figure types ✓")
    return (series,)


@app.cell
def _(PB, graphics, series):
    # The plotly rendering — hover to read the modulus and argument off the surface.
    graphics(
        *series,
        backend=PB,
        grid=False,
        title="plotly: hover reports modulus and argument",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive exploration with widgets

    Because the expression is symbolic, a *parameter* can stay symbolic too. Pass
    `params={sym: (default, min, max)}` and spb builds an ipywidgets (or panel) app that re-lambdifies
    and re-renders on change. Watch the `k`-fold zero at the origin split hues as `k` moves.

    This has no clmmathtools analogue — with a Python callable you would rebuild and redraw by hand.

    Backend note: the widget app embeds the figure in an ipywidgets `Box`, so the figure must itself
    *be* a widget. `PB` (plotly) satisfies that natively via `FigureWidget`. `MB` (matplotlib) only
    does under `%matplotlib widget` (ipympl) — with the default inline/Agg canvas it raises
    `TraitError: ... expected a Widget, not the FigureCanvasAgg`.
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
        grid=False,
        aspect="equal",
        backend=PB,
        imodule="ipywidgets",
        title="z^k (z-1)/(z+1)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap — where each library wins

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
    | Riemann sphere | — | `riemann_sphere_2d` / `_3d` |
    | widgets | — | `params={sym: (init, lo, hi)}` |
    | backends | matplotlib, vedo | matplotlib, plotly, bokeh, k3d, mayavi |
    | data introspection | `phase.colorize()` returns RGB | `series.get_data()` returns `x, y, abs, arg, rgb` |
    | control over the drawing | full — it is your code | `rendering_kw` passthrough, then the series API's edge |

    The short version: **spb is the better tool when the function is symbolic**, which for classical
    complex analysis it usually is — you get exactness, LaTeX, four backends and widgets for free.
    **clmmathtools stays the better tool when it is not** — an iterated map, a numerically-defined
    transform, a function you only have as samples — and when a portrait is one layer of a scene
    you are assembling out of arbitrary primitives.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase Portraits with `sympy_plot_backends` (spb)
    We use spb's **`graphics()` interface** rather than the older `plot_complex()` / `plot()` interface.
    The distinction matters:

    - `plot_complex(expr, range)` is a one-shot command: it builds series *and* a figure, and combining
      different kinds of plot means the awkward `(p1 + p2)` operator.
    - `graphics(*series, **fig_options)` splits those two jobs. Functions like `domain_coloring(...)`
      and `complex_points(...)` are **series factories** — they return a plain `list` of `BaseSeries`
      objects and draw nothing. `graphics()` then takes any mix of them and renders one figure. So a
      domain coloring, a scatter of its zeros, and a vector field compose by list-splatting, and the
      series themselves stay inspectable (`series.get_data()`) for numeric checks.

    ## Scheme correspondence

    clmmathtools names its four Wegert schemes; spb uses single letters from a much longer menu
    (`"a"`–`"o"`, plus `"k+log"`, plus any user callable `f(w) -> (rgb_img, colorscale)`):

    | clmmathtools `scheme` | spb `coloring` | shows |
    |---|---|---|
    | `plane`          | `"a"`          | hue only — the bare phase *(spb default)* |
    | `phase`          | `"d"`          | + iso-phase lines (constant $\arg f$) |
    | `modulus`        | `"c"`          | + iso-modulus lines (constant $\lvert f\rvert$) |
    | `enhanced`       | `"b"`          | **both** → conformal little squares *(clmmathtools default)* |

    Beyond those four, spb also offers stripe colorings (`"e"`–`"h"`), chessboards (`"i"`, `"j"`),
    pure-magnitude greyscale (`"k"`, `"k+log"`), and magnitude-blended variants (`"l"`–`"o"`) that
    brighten toward poles — which is how you tell a zero from a pole twithout counting hue direction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### history
    This is a port of `04_phase_portraits.py` from **clmmathtools** to **spb**
    (`sympy-plot-backends`), so the two libraries can be compared on the same material.

    The single biggest difference is the *input type*:

    - `clmmathtools` takes a **Python callable** `f: ndarray[complex] -> ndarray[complex]`. It samples,
      it colors, it draws. Nothing knows what the function *is*.
    - `spb` takes a **SymPy expression**. The expression is lambdified for sampling, but it is still
      available symbolically — so zeros, poles, orders, residues and derivatives can be computed
      *exactly* and used to annotate the picture, and axis labels come out as rendered LaTeX for free.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
