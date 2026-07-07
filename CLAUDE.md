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

There are no automated tests. "Correctness" in this repo is demonstrated *inside* the notebooks: math identities are verified numerically with `np.allclose(...)` assertions printed inline (e.g. confirming `Ad_{exp(X)} = exp(ad_X)`). When changing math code, preserve and re-run those checks rather than adding a separate test harness.

## Notebook / script pairing (important)

The `LieGroups/*_Lie_Theory.py` files (`SO2`, `SE2`, `SO3`, `SE3`) are **jupytext "percent"-format** mirrors of their `.ipynb` notebooks (cells delimited by `# %%` and `# %% [markdown]`). Each `.py` header declares the pairing with `formats: ipynb,py:percent`, and `jupytext` is a project dependency, so `uv run jupytext --sync <file>.py` round-trips them.

- **Edit the `.py` file**, not the notebook. The `.ipynb` is a large git-LFS blob containing base64-encoded matplotlib output; it's hard to diff and edit directly.
- After editing a `.py`, run `uv run jupytext --sync <file>.py` to update the notebook (code/markdown only — outputs are preserved). Add `--execute` to re-run and refresh the embedded figures.
- These four notebooks build a deliberate ladder of difficulty: **SO(2)** (abelian baseline, trivial bracket/adjoints) → **SE(2)** → **SO(3)** (bracket = cross product, `Ad_R = R`) → **SE(3)** (6-DOF twists, screw motions, 6×6 adjoints). They share a common style: dark-theme `rcParams`, the same color constants, and every adjoint computed two ways (from the definition and a closed form) then asserted equal with `np.allclose`.
- Older notebooks (`finding-primes.ipynb`, `primes-bokeh.ipynb`) have **no paired `.py`** and were authored in Colab — work in the notebook directly there.

## Git LFS

`.gitattributes` routes `*.ipynb` and `*.pdf` through git LFS. A normal `git diff` / `git show` on these shows LFS pointer files, not real content. Use `git lfs` tooling (or open the file) to inspect actual changes. The `LieGroups/*.pdf` files and `eade-lie.md` are reference papers (Eade, Drummond, the Lie-groups-for-roboticists arXiv paper), not generated output — see `LieGroups/README.md` for reading order. Note `LieGroups/README.md` flags that Eade's notes contain errors (e.g. not always distinguishing the group adjoint `Ad` from the algebra adjoint `ad`).


## Helper libraries

started a helper library, `mathviz`, to help to visualize vector functions and
complex functions and other geometric concepts.
this is in the mathviz/ directory

## Conventions seen in the code

- Matplotlib uses a shared dark-theme `rcParams` block and named hex color constants (`BLUE`, `ORANGE`, ...) defined at the top of each script; reuse them for visual consistency.
- Lie-group code represents SE(2) elements as 3×3 homogeneous matrices and `se(2)` algebra elements via fixed basis matrices `e1, e2, e3`; coordinates are extracted from matrix entries (`omega = X[1,0]`, `ux = X[0,2]`, `uy = X[1,2]`). Each derived quantity is given both a from-first-principles computation and an explicit closed-form, then cross-checked with `np.allclose`.

- markdown tables put limitation on how to use TeX/LaTex code as the vertical line character will be interpreted as table delimiters and break the KaTeX rendering. The fix is to replace the vertical "pipe" symbol with either \rvert or \lvert or similar latex symbols.