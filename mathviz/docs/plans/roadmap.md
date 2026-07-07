# mathviz — roadmap

`mathviz` visualizes the 2D / complex plane: draw on it, push it through maps, color it as a Wegert
phase portrait — factoring out the near-duplicate plane-viz helpers previously copied across
`GeometricLinearAlgebra/` and `Geometry/`. See [`architecture.md`](architecture.md) for the design.

## Phases

### ✅ Phase 1 — foundation
Package skeleton (src layout, editable install via `[tool.uv.sources]`), `palette`, backend-neutral
`primitives`, the backend seam (`backends/`), matplotlib + vedo backends for the base primitives, and
the chainable `Plane` facade: `grid / axes / vector / basis / segment / line / curve / points / text /
raster`, with `display()` / `save()`.

### ✅ Phase 2 — maps (the push-forward)
`maps.py`: one operation — push the domain grid/shapes through a callable and draw the image — unifying
`apply_matrix` (linear), `show_operator` (geometric operators), and `apply_complex` (conformal). Clean
pole/branch handling (non-finite → split the drawn curve). Also `vector_field` / `Plane.field` /
`Plane.field_complex`: 2D fields drawn as arrows, **uniform length with magnitude by color** by default
(`normalize=True`, `cmap="viridis"`).

### ✅ Phase 3 — phase portraits (Wegert)
`phase.py`: domain coloring of complex functions. Schemes `plain` / `phase` / `modulus` / `enhanced`
(hue = `arg f`; optional phase and log-modulus contours, equally spaced → conformal cells). Zeros are
where hues meet CCW; poles render white. `Plane.phase_portrait(f, scheme=…)`; `Raster` now supported in
the vedo backend (textured image behind the vector layer).

### ⬜ Phase 4 — 3D scene, analytic landscapes & Riemann surfaces
A `Space3D` scene (perspective camera, 3D primitives) on the vedo backend:
- **Analytic landscape**: the surface `z ↦ |f(z)|` colored by `arg f(z)` (reuses `phase.colorize`).
- **3D vector fields**: `Space3D().field(f)` for `f: ℝ³→ℝ³` (see [`backlog.md`](backlog.md)).
- Riemann-surface scaffolding (patched sheets, branch cuts).

### ⬜ Phase 5 — migrate existing notebooks & polish
- Replace the duplicated `Plane2D` / `new_plot` helpers in `GeometricLinearAlgebra/` and `Geometry/`
  with `import mathviz`; re-verify each notebook renders.
- `animate.py`: parameter sweeps → GIF / inline scrubber (generalizes the shear animation).
- ComplexAnalysis gallery notebooks built on `mathviz` (the original motivation).

## Testing

- `tests/` (pytest, run **from the repo root**): pure logic (palette, primitives, maps, phase, plane,
  backends) plus render smoke tests parametrized over both backends.
- `examples/*.py` (jupytext-paired) double as executable docs **and** feature tests — every cell draws
  something and asserts a property. Regenerate from the repo root with `jupytext --execute`.
