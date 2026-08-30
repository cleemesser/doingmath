# clmmathtools — architecture

## The unifying insight

Three things that look separate across the notebooks are the **same operation**:
a mapping of one space to another - usually a mapping to itself

- `apply_matrix(M)` — push the grid through a **linear** map (linear algebra).
- `show_operator(f)` — push the grid (+ an optional `probe_shape`) through a geometric **operator**
  (project/rotate/shear).
- `apply_complex(g)` — push the grid through a **complex** function (a conformal map).

All are **"push the domain through a callable and draw the image."** And a **phase portrait** is the
raster complement: **"color the domain by a field derived from `f(z)`."** So:

> **clmmathtools = { a 2D plane you draw on } × { push it through any map } × { color it by a complex field }**

with linear algebra and plane geometry falling out as special cases.

## The seam: primitives → backend

A `Plane` does **not** call a plotting library directly. It records backend-neutral **primitives**
(dataclasses in `primitives.py`: `Grid`, `Arrow`, `Segment`, `Polyline`, `Points`, `Text`, `Raster`)
in a `Scene`. A **backend** (`backends/`) walks the scene and renders it. Keeping the scene as plain,
inspectable data is what makes the backends pluggable *and* the geometry unit-testable without a
display (most tests never render — they assert on the recorded primitives).

```
Plane (facade, chainable)  →  [Primitive, …] in a Scene  →  Backend.render(scene)
        maps.py / phase.py append primitives                MatplotlibBackend | VedoBackend
```

## Backends (decision: pluggable matplotlib + vedo)

Phase portraits are raster images, which is matplotlib's home turf; vedo is the repo's 3D choice and
matches the existing `Geometry/` look. Rather than pick one, the seam supports both, selected per
`Plane(backend=…)` or globally via `set_backend`:

- **matplotlib** — 2D, raster-native (`imshow` for phase portraits), native static PNG that embeds and
  renders on GitHub. The default.
- **vedo** (VTK) — 2D parity with `Geometry/`, plus the path to 3D (Phase 4). Renders **offscreen** to a
  static PNG, displayed inline via IPython (so figures survive a headless
  `jupytext --execute`).
  - also supports interactive figures: either through the k3d backend in the
    browser/notebook, or via a separate vtk/ipyvtklink backend as a desktop app

Both render the *same* `Scene`; only the translation to pixels differs. Render tests are parametrized
over both to keep them at parity.
- note currenly text outputs only approximate one another
## Packaging

`clmmathtools/` is an editable sub-package (its own `pyproject.toml`, **src layout** at
`clmmathtools/src/clmmathtools/`) wired into the root project via `[tool.uv.sources]`. The src layout matters:
without it, running from the repo root would import the `clmmathtools/` *directory* as a namespace package
and shadow the installed one. Because the sub-package has its own env, **notebook/pytest tooling is run
from the repo root** (which has the jupyter/test toolchain and the editable `clmmathtools`).

## Conventions

- Colors live once in `palette.py` as `0xRRGGBB` ints, converted per backend (`hexstr` → vedo,
  `rgb01` → matplotlib).
- Drawing methods are **chainable** (`return self`), so scenes read as one fluent expression.
- Non-finite values (poles, branch points) are handled at the source: `maps` splits curves at NaN;
  `phase` renders singular pixels white.
- Vector fields default to **uniform arrow length + magnitude-by-color** (the length never lies in a
  dense field); opt out with `normalize=False` / `cmap=None`.
