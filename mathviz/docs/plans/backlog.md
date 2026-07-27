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
- **SymPy pretty-printing in notebooks** ✅ *(delivered)*: added `mathviz.show_expr(expr, label=None)`,
  which renders via the notebook's native **LaTeX/MathJax** (`IPython.display` of anything with
  `_repr_latex_`, e.g. SymPy matrices) when `mv.in_notebook()`, and falls back to `sympy.pretty` text
  when headless. Swept the sympy-heavy GLA notebooks (`00`, `06`, `07`) to use it for matrix/expression
  displays (`sp.pprint` → `mv.show_expr`), keeping the surrounding `print("label")` captions and the
  boolean/verification prints. Verified: 07/06/00 now emit `text/latex` outputs (7/14/5), no errors.
  *Possible follow-up:* also convert inline `print("x =", expr)` expression prints to `show_expr`.
- **`animate.py`** (Phase 5) ✅ *(delivered)*: parameter sweeps over `build(t) -> Plane | Space3D` for
  `t ∈ [0,1]`. `animate.to_png(scene)` renders a scene to PNG **bytes** offscreen (both backends'
  `save` now accept a file object); `render_frames` pre-renders the sweep; `mv.scrubber(build)` drives
  cached frames with an ipywidgets `Play`/slider (instant scrubbing, no live 3D backend needed);
  `mv.to_gif(build, path)` writes an animated GIF via pillow — generalizing the hand-rolled shear
  animation in `Geometry/2dplaneVedo.py`. `ping_pong=True` loops 0→1→0 seamlessly.
  **Matrix paths** `mv.matrix_path(M, kind)` give `t ↦ M(t)` with `M(0)=I`, `M(1)=M`:
  `lerp` (straight line in matrix space — passes through `det = 0` on any half-turn, kept as the
  teaching counterexample), `polar` (SVD `M = R·P`; turn `R`, raise `P` to the power `t`), and
  `geodesic` (`exp(t·log M)`, the one-parameter subgroup). `mv.animate_matrix(M, kind=…)` is the
  convenience wrapper. Example: `examples/interactive_experiment1.ipynb`; tests: `tests/test_animate.py`.
  *Gotcha encoded in `animate.real_logm`:* `scipy.linalg.logm` returns the **principal** log, whose
  branch cut is the negative real axis — so `logm(-I) = iπI` (complex) and a naive `expm(t·logm(M))`
  collapses exactly like the lerp. A real log exists iff every negative eigenvalue has an *even*
  number of Jordan blocks; `-I` qualifies (generator `[[0,-π],[π,0]]`), `diag(-1,-2)` does not.
  `scipy`/`ipywidgets` are lazily imported and declared under the `interactive` optional extra.
  *Remaining polish:* paths start at `I` only (no general `A → B`); 2×2 only.

  **Fundamental operations** (`maps.rotation/scaling/shear/reflection/projection`): GL(2, ℝ) has two
  connected components split by the `det = 0` wall. Rotation / scaling / squeeze `diag(k,1/k)` / shear
  are in the identity's component (`polar`, `geodesic` both work); a **reflection** (`det < 0`) is in
  the other component and a **projection** (`det = 0`) is on the wall — only `lerp` renders them, and
  the collapse is the mathematics. *Two bugs found and fixed while adding these:* `polar_path` produced
  a silently **discontinuous** path for singular `M` (`σ**t` jumps from `1` at `t=0` to `0` for all
  `t>0`), and `real_logm`'s `det <= 0` guard was an exact sign test that `det(projection(π/6)) = 1.6e-17`
  slipped straight through. Both now share `_reject_unreachable`, which tests rank via singular values
  (`σ_min <= 1e-10·σ_max`) — scale-invariant, and correct in principle since `det exp(A) = e^{tr A} > 0`
  strictly means *no* singular matrix has a logarithm.

  **Two-parameter composition** — `mv.scrubber2(build)` renders an `n1 × n2` grid of frames and gives
  one Play/slider per axis. Used to show `R·S ≠ S·R`, with the Lie bracket `[log R, log S]` as the
  infinitesimal obstruction (verified against `e^{εA}e^{εB}e^{-εA}e^{-εB} = I + ε²[A,B] + O(ε³)`), and
  to split the polar decomposition `M = R·P` into a rotate slider and a stretch slider.

  **Holomorphic sweeps** — `mv.homotopy(g)` gives `g_t(z) = (1-t)z + t·g(z)`, holomorphic at every `t`
  (convex combination), used for `z²`, `1/z`, `exp z`, Joukowski `z + 1/z`; the moving critical point
  where `g_t' = 0` (and conformality dies) is marked. `mv.mobius(A)` / `mv.mobius_path(A)` /
  `animate.classify_mobius(A)` do Möbius maps as one-parameter subgroups of PSL(2, ℂ), classified by
  `tr²` into elliptic / parabolic / hyperbolic / loxodromic — the exact complex mirror of
  rotation / shear / squeeze / rotate-and-scale.
- **Migrate existing notebooks** (Phase 5): delete the ~10 duplicated `Plane2D` / `new_plot` helpers in
  `GeometricLinearAlgebra/` and `Geometry/`; replace with `import mathviz`; re-verify renders.
- **Phase-portrait anti-aliasing** ✅ *(delivered)*: contour bands looked jagged for **two** independent
  reasons. (1) `Plane.phase_portrait` let `res` default below the pixel width the raster is stretched
  across — it now defaults to `max(view.size)`. (2) The brightness ramp `_sawtooth` *resets
  discontinuously* at each band edge, and point-sampling a step function aliases into a staircase at
  **any** `res`; `interpolation="bilinear"` in the mpl backend cannot recover an edge whose sub-pixel
  position was never measured. Fixed by **box-filtering the ramp exactly** over each pixel
  (`aa=True`, default): `∫₀ˣ frac = ⌊x⌋/2 + frac(x)²/2` gives a closed form that → `frac(t)` as the
  band width → 0 and → `1/2` as it → ∞, so sub-pixel contours fade to flat tone instead of moiré.
  The band width comes from `|f'/f|` — **shared by both contour families**, since `d(log f) = (f'/f)dz`
  splits into `d log|f|` (real) and `d arg f` (imaginary), which is also why `enhanced` cells are
  square — and `f' = ∂f/∂x` for holomorphic `f`, so one `np.gradient` of the existing samples recovers
  it with **zero extra evaluations of `f`**. Costs ~30% time (18.5 ms vs 14.0 ms at `res=600`).
  *Measured against 3× supersampling:* SSAA needs 9× the evaluations, leaves brightness noise
  unchanged (std 0.112 vs 0.118) and drifts the mean (0.577 vs the true 0.681 = 0.825²); analytic AA
  gives std 0.011 and the exact mean. SSAA *does* smooth hue, which analytic AA leaves alone — only
  visible near essential singularities. `aa=False` restores the old look.

  *Follow-up — `Space3D` was left out.* `landscape` / `riemann_root` / `riemann_log` call `colorize`
  directly, so they never got the new AA. Plumbed through (`aa=True` default). Doing so exposed that
  the original `_d_logf` **silently assumed a holomorphic `(x, y)` sample grid**: it took only
  `∂/∂axis1` and leaned on `b = i·a` to make one width serve both contour families. True for
  `landscape`, false for the Riemann surfaces, which sample `(ρ, φ)` and `(u, v)` — on `riemann_log`'s
  grid, axis 1 moves `log z` purely imaginarily (phase only) and axis 0 purely really (modulus only),
  so the old formula would over-blur the modulus contours by **1.44×**. Replaced by `_d_logw(w) ->
  (a, b)` (both axes), with `width_phase = hypot(Im a, Im b)` and `width_mod = hypot(Re a, Re b)`.
  On a holomorphic grid `b = i·a` and both collapse back to `|f'/f|`, so **the 2D portraits are
  unchanged** (99.9th pct Δrgb = 5e-4; the max is at poles where `np.gradient` is one-sided).
  Also fixed: `landscape(**kw)` used to branch to `colorize(w, **kw)` and *silently drop `scheme`*;
  and `_sawtooth` hit `inf - inf` (a RuntimeWarning) at exact zeros where `log|f| = -inf`.
  *Measuring AA in 3D:* the mean of `|Δbrightness|` is a **bad** metric (0.0198 → 0.0180) — AA does not
  lower a step's height, it grades the one cell straddling it. Count hard jumps instead: 557 → 116
  (4.8×) at `res=140`, touching only 4.5% of cells. Mesh coarseness (`res=140` across a 720 px view)
  remains a separate, additive cause of chunkiness — raise `res` for stills.
- **Phase-portrait polish**: alternative color wheels (NIST/Wegert palettes), a `steps`/base control UI,
  and marking detected zeros/poles automatically.
- **Riemann surfaces — further** (Phase 4+): beyond `riemann_root`/`riemann_log`, general algebraic
  surfaces (patched sheets for arbitrary `w(z)`) and automatic branch-cut location.
