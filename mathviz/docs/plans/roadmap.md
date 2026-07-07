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

### ✅ Phase 4 — 3D scene, analytic landscapes & Riemann surfaces
A `Space3D` scene (perspective camera, 3D primitives `Arrow3D`/`Line3D`/`Points3D`/`Surface`), rendered
on **both** backends (vedo VTK meshes, matplotlib mplot3d):
- **Analytic landscape**: `Space3D.landscape(f)` — the surface `|f(z)|` colored by `arg f(z)` (reuses
  `phase.colorize`), height clipped to `zmax` (or `log=True`).
- **3D vector fields**: `Space3D.field(f)` for `f: ℝ³→ℝ³` (3×3 matrix or point-map); uniform arrow
  length, magnitude by color — the 2D convention carried into 3D.
- **Riemann surfaces**: `Space3D.riemann_root(n)` (√z / z^{1/n}, sheets joined at the branch point) and
  `Space3D.riemann_log()` (the log helicoid), parametrized by the value `w` and colored by phase.

### 🚧 Phase 5 — migrate existing notebooks & polish
- **`GeometricLinearAlgebra/`**: ✅ migrated `00, 01, 02, 05, 06, 07` — the inline `Plane2D` (vedo)
  and `new_plot`/k3d toolkits replaced by `mathviz` (thin adapters keep the original call sites in the
  k3d ones). ⬜ `03, 04` are pending — they draw 3D **filled** parallelograms/parallelepipeds
  (`k3d.mesh`), so they need mathviz to grow 2D/3D **polygon/mesh** primitives first.
- ⬜ **`Geometry/`** notebooks.
- Surfaced a library fix along the way: the matplotlib backend is now **pyplot-free** (Figure + Agg
  canvas), so it emits static PNGs even when a notebook's kernel has the ipympl widget backend active.
- ⬜ `animate.py`: parameter sweeps → GIF / inline scrubber (generalizes the shear animation).
- ⬜ ComplexAnalysis gallery notebooks built on `mathviz` (the original motivation).

## Testing

- `tests/` (pytest, run **from the repo root**): pure logic (palette, primitives, maps, phase, plane,
  backends) plus render smoke tests parametrized over both backends.
- `examples/*.py` (jupytext-paired) double as executable docs **and** feature tests — every cell draws
  something and asserts a property. Regenerate from the repo root with `jupytext --execute`.
