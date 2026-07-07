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
mv.set_backend("mpl")          # or "vedo"; or Plane(backend="vedo")
mv.Plane(extent=3).grid().basis().curve(circle).display()
```

## Status

Phase 1 (this commit): package skeleton, palette, primitives, the backend seam, matplotlib + vedo
backends for the base primitives (grid/axes/vector/basis/segment/curve/points/text), and the `Plane`
facade. Maps, phase portraits, and 3D landscapes follow.
