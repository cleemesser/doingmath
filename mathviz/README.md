# mathviz

A small library for visualizing the 2D / complex plane: draw vectors, grids, and curves; push the
plane through a **map** (a 2×2 matrix, a linear operator, or an arbitrary complex function) to see its
image; and color the plane by a complex field as a **Wegert phase portrait**.

It factors out the near-duplicate "2D plane" helpers previously copied across `GeometricLinearAlgebra/`
and `Geometry/`, and generalizes them: `apply_matrix` (linear algebra), `show_operator` (geometric
operators), and conformal-map images (complex analysis) are all the *same* operation — push the domain
through a callable and draw the image — while phase portraits are the raster complement (color the
domain by `arg f(z)`).

## Backends

Drawing is backend-neutral. A `Plane` records primitives; a backend renders them:

- **matplotlib** — 2D, raster-native (best for phase portraits), static PNG that embeds and renders on
  GitHub. The default.
- **vedo** (VTK) — 2D parity with the `Geometry/` notebooks plus 3D (analytic landscapes, Riemann
  surfaces). Offscreen static PNG or interactive.

```python
import mathviz as mv
mv.set_backend("mpl")                       # or "vedo"; or Plane(backend="vedo")
mv.Plane(extent=3).basis().curve(circle).display()          # primitives
mv.Plane(extent=3, grid=False).apply_matrix([[1, 1], [0, 1]]).display()   # linear map
mv.Plane(extent=2, grid=False).apply_complex(lambda z: z**2).display()    # conformal map
mv.Plane(extent=3).field([[0, -1], [1, 0]]).display()                     # vector field
mv.Plane(extent=2, grid=False, axes=False).phase_portrait(lambda z: z**2).display()  # phase portrait
```

**Vector fields.** `Plane.field(f)` (a 2×2 matrix or a point-map) and `Plane.field_complex(g)` draw a
field as arrows. By default arrows are **uniform length with magnitude shown by color** (`normalize=True`,
`cmap="viridis"`); pass `normalize=False` for length-encodes-magnitude, or `cmap=None` for one color.
(2D only for now — 3D fields await the Phase-4 3D scene.)

## Tests and examples

- **Unit + render tests** live in `tests/` and run with pytest **from the repo root** (the root
  project has the jupyter/test toolchain; this sub-package's own env is deliberately minimal):

  ```bash
  uv run pytest            # 45 tests: palette, primitives, maps, plane, backends, and render
  ```

  The `test_render.py` cases are parametrized over both backends and assert a non-blank PNG is
  produced headlessly.

- **Example notebooks** in `examples/` double as executable docs *and* feature tests — every cell
  draws something and asserts a property (determinant = image area, idempotent projection, orthogonal
  rotation, `z²` conformality, `1/z` pole splitting, Möbius invertibility). Regenerate them **from the
  repo root** so the full toolchain is on the path:

  ```bash
  uv run jupytext --to notebook --execute mathviz/examples/01_plane_and_primitives.py
  uv run jupytext --to notebook --execute mathviz/examples/02_maps.py
  ```

## Status

- **Phase 1** — package skeleton, palette, primitives, the backend seam, matplotlib + vedo backends,
  and the `Plane` facade (grid/axes/vector/basis/segment/line/curve/points/text/raster). ✓
- **Phase 2** — `maps.py`: `apply_matrix` (linear), `show_operator` (operators), `apply_complex`
  (conformal), unified as one push-forward, with clean pole/branch handling. ✓
- **Phase 3** — Wegert phase portraits (`plain`/`phase`/`modulus`/`enhanced`) + `Raster` in the vedo
  backend. ✓
- **Phase 4** — 3D scene (`Space3D`, on vedo + mplot3d): analytic landscapes (`landscape`), 3D vector
  fields (`field`), and Riemann surfaces (`riemann_root`, `riemann_log`). ✓
- **Phase 5** — migrate the `GeometricLinearAlgebra/` and `Geometry/` notebooks onto `mathviz`. 🚧
  Done: **all GLA `00`–`07`**. Pending: `Geometry/` notebooks.

See [`docs/plans/`](docs/plans/) for the full roadmap, architecture notes, and backlog.
