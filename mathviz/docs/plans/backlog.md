# mathviz — backlog & reminders

Deferred items and explicit reminders, so nothing is lost between sessions.

## ✅ 3D scene — `Space3D().field(f)`  *(delivered)*

Done in Phase 4. `Space3D` is a perspective 3D scene with `Arrow3D`/`Line3D`/`Points3D`/`Surface`,
rendered on both backends (vedo VTK + matplotlib mplot3d). `Space3D.field(f)` draws a 3D vector field
(3×3 matrix or `(N,3)->(N,3)`) with the 2D convention (uniform length, magnitude by color);
`Space3D.landscape(f)` draws the analytic landscape; `Space3D.riemann_root(n)` / `riemann_log()` draw
multi-sheeted **Riemann surfaces**. Requested by the user 2026-07-07; delivered same day — Phase 4 done.

## Other deferred items

- **Interactive vedo rendering** *(user-requested)*: today the vedo backend only renders **offscreen →
  static PNG**. Add an interactive path:
  - **Detect the environment** — are we in an IPython/Jupyter notebook (`get_ipython()` exists and has a
    kernel) vs. a plain script/headless run? Default to static PNG when headless, offer interactive when
    in a notebook.
  - **Let the caller choose** — e.g. `Plane(..., interactive=True)` / `Space3D(..., interactive=True)` or
    a `display(interactive=...)` flag, so a user can force static or live regardless of detection.
  - **Let the caller choose the vedo display backend** — vedo supports several (`k3d`, `trame`,
    `ipyvtklink`, `2d`, `vtk`); expose it (e.g. `vedo_backend="k3d"`), since the repo already pins the
    trame/vue2 stack for this. Wire it through `vedo.settings.default_backend` and a live `Plotter.show`
    (not offscreen) on the interactive path. Note interactive 3D is where vedo shines over matplotlib.
- **`animate.py`** (Phase 5): parameter sweeps → GIF / inline scrubber; generalize the
  `Geometry/2dplaneVedo.py` shear animation. Pre-render frames offscreen; drive with an ipywidgets
  Play/slider.
- **Migrate existing notebooks** (Phase 5): delete the ~10 duplicated `Plane2D` / `new_plot` helpers in
  `GeometricLinearAlgebra/` and `Geometry/`; replace with `import mathviz`; re-verify renders.
- **Phase-portrait polish**: alternative color wheels (NIST/Wegert palettes), a `steps`/base control UI,
  and marking detected zeros/poles automatically.
- **Riemann surfaces — further** (Phase 4+): beyond `riemann_root`/`riemann_log`, general algebraic
  surfaces (patched sheets for arbitrary `w(z)`) and automatic branch-cut location.
