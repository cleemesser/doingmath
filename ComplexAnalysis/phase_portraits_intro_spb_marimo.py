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

__generated_with = "0.24.2"
app = marimo.App(app_title="Intro to Phase Portraits", auto_download=["ipynb"])


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

    In the code below, we first import the sympy objects we want and activate fancy printing. Then we define the symbol `z` to represent points in the complex plane:
    ```python
    z = symbols("z")
    ```
    This call creates an object in Python that represents a mathematical symbol.

    We then define two domains, `DOMAIN2` and `DOMAIN3`. Each one is a square area of the complex plane.
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
        together,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The four ways to encode the complex "z-plane"
    ### Overview
    1. First write $z$ in polar form: $z= R e^{i\theta}$. Color each point by the angle $\theta$ (Figure 1a).
    2. Build the polar grid by drawing iso-phase lines (Figure 1b).
    3. Add iso-modulus (r) contours (Figure 1c).
    4. Combine 1, 2, and 3 into a standard phase portrait (Figure 1d) with coloring="b"

    ### Encoding the polar angle with a hue
    We can write a number in the complex plane as $z = Re^{i\theta}$, so we can assign a hue from a color scheme to every value of $\theta$ from 0 to $2\pi$. `spb` does this in the code below:
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
    Add "iso-phase" lines, that is, lines where $\theta$ is constant. This starts a polar grid on the complex plane. Remove the tick marks too, so the domain stays implicit.
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

    Series factories return objects, not pictures, so you can assert on the same data the renderer
    sees. `ComplexDomainColoringSeries.get_data()` returns `(x, y, abs, arg, rgb_img, colorscale)`.
    Note that spb also hands back the modulus and argument grids.
    """)
    return


@app.cell
def _(DOMAIN3, domain_coloring, graphics, mo, z):
    # how to enable matplotlib interactive pan/zoom/save
    mo.mpl.interactive(
        graphics(
            *domain_coloring(
                z**2, DOMAIN3, coloring="b", n=2500, colorbar=False
            ),
            grid=False,
            axis=False,
            aspect="equal",
            show=False,
        ).fig
    )
    return


@app.cell
def _(DOMAIN2, DOMAIN3, MB, domain_coloring, graphics, z):
    def portrait(f, domain=DOMAIN3, coloring="b", n=500, **kwargs):
        """One titled domain-coloring panel, titled from the expression itself."""
        series = domain_coloring(
            f, domain, coloring=coloring, n=n, colorbar=False
        )
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
        (series,) = domain_coloring(
            expr, domain, coloring=code, n=n, colorbar=False
        )
        return series.get_data()[4].astype(float)

    return coloring_image, portrait


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

    plotgrid(*[portrait(f) for f in gallery], nr=3, nc=2, size=(8, 12))
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

    results = []
    for backend in (MB, PB, BB):
        try:
            p = graphics(
                *series, backend=backend, grid=False, show=False, aspect="equal"
            )
            results.append(
                f"{backend.__name__:20s} -> {type(p.fig).__module__}"
            )
        except AttributeError as e:
            results.append(
                f"{backend.__name__:20s} -> SKIPPED ({e.__class__.__name__}: {e})"
            )

    print("\n".join(results))
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
    # Phase Portraits with `sympy_plot_backends` (spb)
    We use spb's `graphics()` interface, not the older `plot_complex()` / `plot()` interface. The
    difference matters:

    - `plot_complex(expr, range)` is a one-shot command. It builds the series and the figure
      together. To combine different kinds of plot, you use the awkward `(p1 + p2)` operator.
    - `graphics(*series, **fig_options)` splits those two jobs. Functions like `domain_coloring(...)`
      and `complex_points(...)` are series factories: they return a plain `list` of `BaseSeries`
      objects and draw nothing. `graphics()` then takes any mix of them and renders one figure. A
      domain coloring, a scatter of its zeros, and a vector field therefore compose by unpacking
      lists into one call. The series also stay inspectable (`series.get_data()`) for numeric checks.

    ## Scheme correspondence

    clmmathtools names four Wegert schemes. spb uses single letters from a much longer menu (`"a"`
    to `"o"`, plus `"k+log"`, plus any user callable `f(w) -> (rgb_img, colorscale)`):

    | clmmathtools `scheme` | spb `coloring` | shows |
    |---|---|---|
    | `plane`          | `"a"`          | hue only, that is, the bare phase (spb default) |
    | `phase`          | `"d"`          | plus iso-phase lines (constant $\arg f$) |
    | `modulus`        | `"c"`          | plus iso-modulus lines (constant $\lvert f\rvert$) |
    | `enhanced`       | `"b"`          | both, which draws small conformal squares (clmmathtools default) |

    Beyond those four, spb also offers stripe colorings (`"e"` to `"h"`), chessboards (`"i"`, `"j"`),
    pure-magnitude greyscale (`"k"`, `"k+log"`), and magnitude-blended variants (`"l"` to `"o"`) that
    brighten toward poles. The brightness tells a zero from a pole without counting hue direction.
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
