# Complex Analysis — Table of Contents

Notebooks are jupytext pairs (`.py` ⟷ `.ipynb`, `make` runs `jupytext --sync *.py`).
`.ipynb` files carry the rendered output; edit the `.py`.

## Suggested reading order

| # | Notebook | One-line summary |
|---|---|---|
| 1 | [`02_maps.py`](02_maps.py) | The core operation: push the plane's grid through a map — linear, geometric, complex, Möbius. |
| 2 | [`maps-from-linear-to-complex-holomorphic.py`](maps-from-linear-to-complex-holomorphic.py) | Extended version of (1), plus non-holomorphic maps and a scalar that measures non-conformality. |
| 3 | [`maps-animated.py`](maps-animated.py) | The same tour with every map animated along a one-parameter subgroup; scrubbers + GIF export. |
| 4 | [`04_phase_portraits.py`](04_phase_portraits.py) | Wegert phase portraits in `clmmathtools` — hue = phase, four schemes, zeros/poles by winding. |
| 5 | [`phase_portraits_spb.py`](phase_portraits_spb.py) | Port of (4) to `sympy_plot_backends`, comparing the two libraries on identical material. |
| 6 | [`05_analytic_landscapes_and_3d.py`](05_analytic_landscapes_and_3d.py) | `Space3D`: analytic landscapes, 3D vector fields, Riemann surfaces of √z / log z. |
| 7 | [`escher/01_escher_droste.py`](escher/01_escher_droste.py) | Escher's *Print Gallery* via `z = exp(α·log w)` — the de Smit–Lenstra paper, implemented. |

---

## 1. `02_maps.py` — pushing the plane through maps

*clmmathtools examples 2.* One push-forward serves three worlds; every claim is asserted numerically.

1. Linear maps — `apply_matrix` (image of the unit square has area = |det M|)
2. Geometric operators — `show_operator` (`probe_shape=mv.FLAG`; idempotent projection, orthogonal rotation)
3. Complex functions — `apply_complex` (z² → confocal parabolas, 1/z → circles, pole splits the curve)
4. A Möbius transformation (circles-and-lines → circles-and-lines; invertibility check)

## 2. `maps-from-linear-to-complex-holomorphic.py` — from linear maps to analytic functions

Same sections 1–4 as above, then:

5. Non-holomorphic maps — when the grid is *not* conformal (`z̄·z = |z|²` folds to a line; the twist `f(r,θ) = (r, θ+k·r)` spirals: area kept, angles not)
   - Measuring non-conformality: the "similarity deviation"

## 3. `maps-animated.py` — the same tour, animated

`animate` pre-renders frames offscreen to PNG; ipywidgets `Play`/slider scrubs the cache. Lie-group
first: `M(t) = exp(t·log M)` where a logarithm exists, `lerp`/holomorphic `homotopy` where it does not.

1. Linear maps — the geodesic one-parameter subgroup (`GL⁺(2,ℝ)`, `M(s)M(t) = M(s+t)`)
2. Operators — rotation is a geodesic; projection cannot be (`det = 0` ⇒ no logarithm)
3. Complex functions — `z²` and `1/z` are *not* Möbius, so no geodesic
4. Möbius — the `PSL(2,ℂ)` geodesic (`mobius_path(A)`)
5. Non-holomorphic maps — the geodesic does not reach them either
- Portable GIF export
- *Open Todo list in the header: text for learners, scrubber sizing (400×400 @ higher pixel ratio).*

## 4. `04_phase_portraits.py` — phase portraits (Wegert)

Hue = `arg f(z)`; contours encode more. Following Wegert, *Visual Complex Functions* (2012).

- The four schemes on `f(z) = z`: `plane` (hue) / `phase` / `modulus` / `enhanced` (default, conformal squares)
- A small gallery (enhanced): z², 1/z, (z²−1)/(z²+1), exp z, sin z
- Reading the portrait: winding number = order of a zero/pole (CCW = zero, CW = pole, poles render white)
- Both backends: matplotlib and vedo (grid composes over the raster)

## 5. `phase_portraits_spb.py` — phase portraits with `sympy_plot_backends`

Port of (4) for a library comparison, using `graphics(*series)` rather than one-shot `plot_complex()`.

- Scheme correspondence: `plane`↔`"a"`, `phase`↔`"d"`, `modulus`↔`"c"`, `enhanced`↔`"b"` (spb has 16 + user callables)
- The four ways to encode the z-plane (polar hue, iso-phase, iso-modulus, all)
- Checking the schemes numerically (`series.get_data()` → x, y, abs, arg, rgb)
- A small gallery (enhanced) — LaTeX labels lifted from the SymPy expression
- Reading the portrait: winding number = order of a zero/pole
- Where `graphics()` pays off: composing series (coloring + zero scatter + vector field)
- Things spb gives you that `Plane` does not
  - The analytic landscape from the same expression
  - The Riemann sphere, stereographically projected
  - Magnitude-blended colorings tell zeros from poles
- Backends: the same series, four renderers (matplotlib, plotly, bokeh, k3d)
- Interactive exploration with widgets (`params={sym: (init, lo, hi)}`)
- Recap — where each library wins (comparison table; symbolic → spb, numeric/compositional → clmmathtools)

## 6. `05_analytic_landscapes_and_3d.py` — the 3D scene

`Space3D`, the perspective sibling of `Plane`, on vedo (default) or matplotlib.

- Analytic landscapes — `Space3D.landscape(f)`: surface `|f|` colored by `arg f`; valleys = zeros, clipped towers = poles
- 3D vector fields — `Space3D.field(f)`: uniform-length arrows, magnitude by color (matrix or callable)
- Riemann surfaces — `riemann_root(n)` (√z parking ramp), `riemann_log()` (infinite helicoid)
- Same scene, either backend (vedo/VTK vs mplot3d)

## 7. `escher/01_escher_droste.py` — the logarithm of an image (*Print Gallery*)

After 3Blue1Brown's video and de Smit & Lenstra, *Notices AMS* **50** (2003) 446–451. Everything
follows from `z = np.exp(ALPHA * np.log(w))` with `α = (2πi + log 256) / 2πi`. Uses a synthetic
self-similar picture, not the copyrighted lithograph.

1. The constants (period 256; the four studies repeat under ×4 and ×i)
   - Why that α forces that γ — one line of algebra
2. The straight world: an ordinary drawing that contains itself (`f(256z) = f(z)`)
3. Taking the logarithm (paper Fig. 14) — the lattices (Fig. 10)
4. The map — and the lithograph (`g = f ∘ h`, `h(w) = w^α`)
5. The grid — where Escher's headaches were
   - 5a. Escher's two false starts (Figs. 2, 3) — measuring "square appearance"
   - 5b. The mechanism: `exp` of a lattice, tilted
   - 5c. Escher's grid proper: the square grid pulled back (Figs. 4, 6)
6. Loop transport — how γ was *measured* (Figs. 7, 8, 9): winding number, loop ABCDA, the measurement
7. Rotation and replication — the γ symmetry: the replication group (fourth roots of γ), Fig. 15
   magnification triptych, replication as motion (the Droste zoom)
8. Variants: other δ (the paper's closing paragraph)
9. Using your own image
10. The rest of the paper — what each remaining figure would take (done / needs a fit / blocked on artwork)
- Recap

---

## Reference material in this directory

| Path | What it is |
|---|---|
| `Elias_Wegert_Visual_Complex_Functions_An_Intro.pdf` / `.md` / `_images/` | Wegert's *Visual Complex Functions* — source for notebooks 4 and 5 |
| `17_Wegert2016_Chapter_VisualExplorationOfComplexFunctions.pdf` | shorter Wegert chapter on visual exploration |
| `escher/2003-de_smit-lenstra-escher.pdf` / `.md` / `_images/` | de Smit & Lenstra paper + extracted figures — source for notebook 7 |
| `escher/How (and why) to take a logarithm of an image [ldxFjLJ3rVY].webm` | the 3Blue1Brown video |
| `escher/README.md` | links for the Escher material |
