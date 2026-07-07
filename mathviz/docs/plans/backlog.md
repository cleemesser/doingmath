# mathviz — backlog & reminders

Deferred items and explicit reminders, so nothing is lost between sessions.

## 🔔 3D scene — `Space3D().field(f)`  *(user-requested reminder)*

The current `Plane` is a **top-down 2D** scene, so `field` / `field_complex` are 2D only. A 3D vector
field needs a new **`Space3D`** scene:

- Perspective (not orthographic) camera; 3D primitives (`Arrow3D`, `Line3D`, `Surface`, `Points3D`).
- `Space3D().field(f)` for `f: ℝ³ → ℝ³`, sampling a 3D lattice and drawing arrows — same default
  convention as 2D (**uniform length, magnitude by color**).
- Rides on the **vedo** backend (native 3D); matplotlib `mplot3d` as a possible fallback.
- Naturally shares the backend seam: add 3D primitive dataclasses; `VedoBackend` grows a 3D render
  path; a `Space3D` facade mirrors `Plane`.

This lands in **Phase 4** alongside analytic landscapes (the surface `|f(z)|` colored by `arg f(z)`),
since both need the 3D scene. Raised by the user on 2026-07-07: *"remind me later about the 3D scene
`Space3D().field(f)` implementation."*

## Other deferred items

- **`animate.py`** (Phase 5): parameter sweeps → GIF / inline scrubber; generalize the
  `Geometry/2dplaneVedo.py` shear animation. Pre-render frames offscreen; drive with an ipywidgets
  Play/slider.
- **Migrate existing notebooks** (Phase 5): delete the ~10 duplicated `Plane2D` / `new_plot` helpers in
  `GeometricLinearAlgebra/` and `Geometry/`; replace with `import mathviz`; re-verify renders.
- **Phase-portrait polish**: alternative color wheels (NIST/Wegert palettes), a `steps`/base control UI,
  and marking detected zeros/poles automatically.
- **Riemann surfaces** (Phase 4+): draw phase portraits on embedded surfaces; locate branch cuts.
