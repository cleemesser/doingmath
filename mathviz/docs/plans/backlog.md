# mathviz — backlog & reminders

Deferred items and explicit reminders, so nothing is lost between sessions.

## ✅ 3D scene — `Space3D().field(f)`  *(delivered)*

Done in Phase 4. `Space3D` is a perspective 3D scene with `Arrow3D`/`Line3D`/`Points3D`/`Surface`,
rendered on both backends (vedo VTK + matplotlib mplot3d). `Space3D.field(f)` draws a 3D vector field
(3×3 matrix or `(N,3)->(N,3)`) with the 2D convention (uniform length, magnitude by color);
`Space3D.landscape(f)` draws the analytic landscape; `Space3D.riemann_root(n)` / `riemann_log()` draw
multi-sheeted **Riemann surfaces**. Requested by the user 2026-07-07; delivered same day — Phase 4 done.

## Other deferred items

- **Interactive vedo rendering** ✅ *(delivered)*: the vedo backend now supports a **live widget** path
  alongside offscreen static PNG. `mv.in_notebook()` detects a Jupyter kernel; `display(interactive=…,
  vedo_display=…)` overrides per call; `mv.set_interactive(True|False|"auto")` and
  `mv.set_vedo_display("k3d"|"trame"|…)` set global defaults. `"auto"` → live in a notebook, static when
  headless. **Two live paths:** in a **notebook** → an inline `vedo.settings.default_backend` widget
  (k3d/trame) returned for display; in a **plain script** → a native, orbitable VTK window (blocking
  `Plotter.show(interactive=True)`). Examples: `examples/06_interactive.ipynb` (widgets) and
  `examples/interactive_script.py` (native window). *Remaining polish:* interactive matplotlib (ipympl)
  is still static-only; the notebook 2D live path drops the parallel-projection camera (k3d ignores
  vtkCamera) so live 2D auto-fits instead of holding scale.
- **SymPy pretty-printing in notebooks** *(user-requested)*: the sympy-heavy GLA notebooks (`00`, `06`,
  `07`) use `print(...)` / `sp.pprint(...)`, which emit plain monospace text — they don't use Jupyter's
  native LaTeX/MathJax rendering of expressions. Detect when running in a notebook (an IPython kernel is
  present) and render sympy expressions with `IPython.display.display` (or a `%`-free `init_printing`
  path) instead of `print`, falling back to text when headless. Likely a tiny shared helper,
  e.g. `show_expr(expr, label=...)` that displays LaTeX in a notebook and prints otherwise; then sweep
  the sympy notebooks to use it for matrices/identities. (Not a `mathviz`-core feature — a notebook
  authoring utility — but tracked here with the rest of the plan.)
- **`animate.py`** (Phase 5): parameter sweeps → GIF / inline scrubber; generalize the
  `Geometry/2dplaneVedo.py` shear animation. Pre-render frames offscreen; drive with an ipywidgets
  Play/slider.
- **Migrate existing notebooks** (Phase 5): delete the ~10 duplicated `Plane2D` / `new_plot` helpers in
  `GeometricLinearAlgebra/` and `Geometry/`; replace with `import mathviz`; re-verify renders.
- **Phase-portrait polish**: alternative color wheels (NIST/Wegert palettes), a `steps`/base control UI,
  and marking detected zeros/poles automatically.
- **Riemann surfaces — further** (Phase 4+): beyond `riemann_root`/`riemann_log`, general algebraic
  surfaces (patched sheets for arbitrary `w(z)`) and automatic branch-cut location.
