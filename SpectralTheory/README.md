# Spectral theory

- `gelfand_formula.typ` — Gelfand's spectral radius formula, ρ(A) = lim ‖Aᵏ‖^{1/k}, worked through
  examples in order of difficulty: normal matrices, Jordan blocks, non-normal diagonalizable
  matrices, the choice of norm, Fekete's subadditivity lemma, Gelfand's resolvent proof (with a
  pseudospectrum plot), estimating ρ by repeated squaring, and Lyapunov/Householder norms.
- `gelfand_formula_quarto.qmd` — the same notebook in Quarto, for comparison. Instead of
  `print()`, every result is converted to LaTeX with sympy (`sp.Float`, `sp.Matrix`, symbolic
  closed forms such as Jᵏ and Sᵏ) and placed in the prose: inline Python expressions for single
  values, and Markdown tables or display equations returned from cells.

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

### Quarto version

```bash
uv run quarto render SpectralTheory/gelfand_formula_quarto.qmd            # HTML + Typst PDF
uv run quarto render SpectralTheory/gelfand_formula_quarto.qmd --to html
```

Needs Quarto ≥ 1.4 (for inline `{python}` expressions; tested with 1.10) and `sympy`, which is
already a project dependency. The `.qmd` deliberately has a different stem from the Calepin file:
Quarto's Typst output writes `<stem>.typ` next to the source, which would otherwise overwrite
`gelfand_formula.typ`. Rendered outputs are git-ignored.

Two pandoc quirks shaped the math source. Norms are written `\lVert … \rVert` rather than `\|`, so
they survive inside Markdown tables. And pandoc's TeX→Typst writer drops the space after a
superscript before `\lVert`, so `\lvert\lambda\rvert^k \lVert v \rVert` breaks the Typst PDF;
write `\lvert\lambda^k\rvert \lVert v \rVert` instead.
