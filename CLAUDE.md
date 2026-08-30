# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal collection of self-contained explorations that illustrate math concepts in code (primes, Lie groups). Each topic is a standalone Jupyter notebook (and, for newer work, a jupytext-paired Python script). There is no shared library, package, build artifact, or test suite — notebooks are the deliverable.

## Environment & commands

Dependencies are managed with **uv** (`uv.lock`, `pyproject.toml`); Python **3.14** is required (`.python-version`).

```bash
uv sync                       # install/refresh the environment from uv.lock
uv run jupyter lab            # launch Jupyter to run notebooks
uv run ruff format .          # format Python (ruff is the formatter in use; see .ruff_cache)
uv run jupytext --sync <file>.py            # keep a paired .py <-> .ipynb in sync (code only)
uv run jupytext --sync --execute <file>.py  # sync AND re-run, embedding fresh outputs/figures
MPLBACKEND=Agg uv run python <file>.py      # run a percent-format script headless to check numerics
```

A committed `.envrc` (direnv) syncs the venv and puts `.venv/bin` on `PATH` on `cd`, so on a
machine with direnv enabled (`direnv allow` once) the `uv run` prefix above is optional. It also
watches `pyproject.toml`/`uv.lock` and re-syncs when they change. Keep using `uv run` in anything
written down — scripts, docs, CI — since direnv is a local convenience, not a guarantee.

There are no automated tests. "Correctness" in this repo is demonstrated *inside* the notebooks: math identities are verified numerically with `np.allclose(...)` assertions printed inline (e.g. confirming `Ad_{exp(X)} = exp(ad_X)`). When changing math code, preserve and re-run those checks rather than adding a separate test harness.

## PyTorch (optional, backend-selected)

`torch` is **not** a base dependency — only `Geometry/kingdon-with-pytorch.ipynb` and friends need
it, and the CUDA wheels are ~2.5 GB. It lives behind three mutually exclusive extras, declared in
`[tool.uv] conflicts` so uv rejects any two at once:

```bash
uv sync --extra torch    # normal case: auto-selects the build for this machine
uv sync --extra cpu      # override: force the CPU-only build
uv sync --extra cu129    # override: force CUDA 12.9
```

Non-obvious points, all encoded in `pyproject.toml` comments:

- **There is no separate "MPS wheel."** On Apple silicon the ordinary PyPI `arm64` wheel is already
  built with the MPS backend. So the `torch` extra's `[tool.uv.sources]` entry carries both an
  `extra` and a `marker = "sys_platform != 'darwin'"`: off macOS it points at the CUDA index, and on
  macOS *nothing matches*, letting it fall through to PyPI. The fall-through is the mechanism —
  don't "fix" it by adding a darwin entry.
- Both `[[tool.uv.index]]` blocks are `explicit = true`, so the PyTorch indexes are consulted only
  for `torch` itself. Without that, uv would resolve every package against them and silently pull
  stale mirrors of common deps (jinja2, networkx, sympy).
- `uv sync --extra cu129` on a Mac fails at *install* time, not lock time — that's intended. The
  lockfile stays portable, carrying all three variants (`2.13.0`, `2.13.0+cpu`, `2.13.0+cu129`).
- **`UV_EXTRA` does not exist.** Extras have no environment-variable form (unlike dependency groups,
  which have `UV_DEFAULT_GROUPS`), and uv *silently ignores* the unknown var rather than erroring —
  so `UV_EXTRA=torch uv sync` quietly uninstalls torch. That is why `.envrc` runs
  `uv sync --extra torch` as a command instead of exporting anything. Likewise `--torch-backend=auto`
  exists only on the `uv pip` interface, not `uv sync`/`uv lock`.

Verify the backend is live with `torch.backends.mps.is_available()`, not just `is_built()`.

## Notebook / script pairing (important)

The `LieGroups/*_Lie_Theory.py` files (`SO2`, `SE2`, `SO3`, `SE3`) are **jupytext "percent"-format** mirrors of their `.ipynb` notebooks (cells delimited by `# %%` and `# %% [markdown]`). Each `.py` header declares the pairing with `formats: ipynb,py:percent`, and `jupytext` is a project dependency, so `uv run jupytext --sync <file>.py` round-trips them.

- **Edit the `.py` file**, not the notebook. The `.ipynb` is a large git-LFS blob containing base64-encoded matplotlib output; it's hard to diff and edit directly.
- After editing a `.py`, run `uv run jupytext --sync <file>.py` to update the notebook (code/markdown only — outputs are preserved). Add `--execute` to re-run and refresh the embedded figures.
- These four notebooks build a deliberate ladder of difficulty: **SO(2)** (abelian baseline, trivial bracket/adjoints) → **SE(2)** → **SO(3)** (bracket = cross product, `Ad_R = R`) → **SE(3)** (6-DOF twists, screw motions, 6×6 adjoints). They share a common style: dark-theme `rcParams`, the same color constants, and every adjoint computed two ways (from the definition and a closed form) then asserted equal with `np.allclose`.
- Older notebooks (`finding-primes.ipynb`, `primes-bokeh.ipynb`) have **no paired `.py`** and were authored in Colab — work in the notebook directly there.

## Git LFS

`.gitattributes` routes `*.ipynb` and `*.pdf` through git LFS. A normal `git diff` / `git show` on these shows LFS pointer files, not real content. Use `git lfs` tooling (or open the file) to inspect actual changes. The `LieGroups/*.pdf` files and `eade-lie.md` are reference papers (Eade, Drummond, the Lie-groups-for-roboticists arXiv paper), not generated output — see `LieGroups/README.md` for reading order. Note `LieGroups/README.md` flags that Eade's notes contain errors (e.g. not always distinguishing the group adjoint `Ad` from the algebra adjoint `ad`).


## Helper libraries

started a helper library, `clmmathtools`, to help to visualize vector functions and
complex functions and other geometric concepts.
this is in the clmmathtools/ directory

## Conventions seen in the code

- Matplotlib uses a shared dark-theme `rcParams` block and named hex color constants (`BLUE`, `ORANGE`, ...) defined at the top of each script; reuse them for visual consistency.
- Lie-group code represents SE(2) elements as 3×3 homogeneous matrices and `se(2)` algebra elements via fixed basis matrices `e1, e2, e3`; coordinates are extracted from matrix entries (`omega = X[1,0]`, `ux = X[0,2]`, `uy = X[1,2]`). Each derived quantity is given both a from-first-principles computation and an explicit closed-form, then cross-checked with `np.allclose`.

- markdown tables put limitation on how to use TeX/LaTex code as the vertical line character will be interpreted as table delimiters and break the KaTeX rendering. The fix is to replace the vertical "pipe" symbol with either \rvert or \lvert or similar latex symbols.

- LaTeX built from **adjacent Python string literals** (the usual shape of a `TangleLatex(latex=(...))` block in `GeometricLinearAlgebra/*_marimo.py`) is concatenated with *nothing* between the pieces. TeX scans a control word greedily until a non-letter, so a literal ending in `\qquad` followed by one starting with a letter yields `\qquadv` — an undefined control sequence and a KaTeX parse error. Keep the separating space **inside** a literal: write `r" \qquad "` on its own line, not `r"\qquad"`. Only a following *letter* triggers it (`\qquad` + `\alpha` or `+ 2` tokenize fine), so grep alone can't decide a site is safe — you have to look at the first character of the next literal.