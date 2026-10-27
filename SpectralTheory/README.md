# Spectral theory

- `gelfand_formula.typ` — Gelfand's spectral radius formula, ρ(A) = lim ‖Aᵏ‖^{1/k}, worked through
  examples in order of difficulty: normal matrices, Jordan blocks, non-normal diagonalizable
  matrices, the choice of norm, Fekete's subadditivity lemma, Gelfand's resolvent proof (with a
  pseudospectrum plot), estimating ρ by repeated squaring, and Lyapunov/Householder norms.

## Building

The notebook is a [Typst](https://typst.app) document executed by
[Calepin](https://typst.app/universe/package/calepin), which runs the ```` ```python ```` blocks
in one persistent Python session and splices their output and figures into the page. Calepin
0.0.61 needs the `typst` CLI ≥ 0.15.0 on `PATH`.

```bash
cargo install typst-cli --locked   # or: brew install typst
cargo install calepin --locked
uv run calepin compile SpectralTheory/gelfand_formula.typ             # -> gelfand_formula.pdf
uv run calepin compile SpectralTheory/gelfand_formula.typ --format html
```

Run it through `uv run` so that the `python3` Calepin starts is the project venv (numpy, scipy,
matplotlib). Alternatively set `CALEPIN_PYTHON=.venv/bin/python`. Plain `typst compile` renders
the code without executing it.
