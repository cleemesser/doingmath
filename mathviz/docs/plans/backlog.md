# mathviz — backlog & reminders

Deferred items and explicit reminders, so nothing is lost between sessions.

## ✅ 3D scene — `Space3D().field(f)`  *(delivered)*

Done in Phase 4. `Space3D` is a perspective 3D scene with `Arrow3D`/`Line3D`/`Points3D`/`Surface`,
rendered on both backends (vedo VTK + matplotlib mplot3d). `Space3D.field(f)` draws a 3D vector field
(3×3 matrix or `(N,3)->(N,3)`) with the 2D convention (uniform length, magnitude by color);
`Space3D.landscape(f)` draws the analytic landscape. Requested by the user 2026-07-07; delivered same
day. Remaining 3D work: **Riemann surfaces** (patched sheets / branch cuts).

## Other deferred items

- **`animate.py`** (Phase 5): parameter sweeps → GIF / inline scrubber; generalize the
  `Geometry/2dplaneVedo.py` shear animation. Pre-render frames offscreen; drive with an ipywidgets
  Play/slider.
- **Migrate existing notebooks** (Phase 5): delete the ~10 duplicated `Plane2D` / `new_plot` helpers in
  `GeometricLinearAlgebra/` and `Geometry/`; replace with `import mathviz`; re-verify renders.
- **Phase-portrait polish**: alternative color wheels (NIST/Wegert palettes), a `steps`/base control UI,
  and marking detected zeros/poles automatically.
- **Riemann surfaces** (Phase 4+): draw phase portraits on embedded surfaces; locate branch cuts.
