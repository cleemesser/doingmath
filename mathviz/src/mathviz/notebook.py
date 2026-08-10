"""Notebook-friendly display of expressions.
`show_md` renders full markdown which can be used with rf"$... {sympy.latex(expr)$"}

`show_expr` renders a value with the notebook's **native LaTeX** (MathJax) when running in Jupyter —
so a SymPy matrix or identity appears as real math instead of monospace text — and falls back to a
pretty text print when headless (script / CI), so nothing breaks either way. It reuses
:func:`mathviz.in_notebook` for the detection.
"""

from __future__ import annotations

from . import backends


def show_md(*args, **kwargs):
    if backends.in_notebook():
        from IPython.display import display, Markdown

        return display(Markdown(*args, **kwargs))
    else:
        try:
            import sympy as sp

            text = sp.pretty(expr)
        except Exception:
            text = str(expr)
        print(text)


def show_expr(expr, label=None):
    """Display ``expr`` as rendered math in a notebook, or pretty-printed text when headless.

    In a notebook, objects with a ``_repr_latex_`` (SymPy expressions/matrices) render as MathJax; a
    ``label`` string, if given, is prepended as ``label = <expr>``. Returns ``None``.
    """
    if backends.in_notebook():
        from IPython.display import Math, display

        latexer = getattr(expr, "_repr_latex_", None)
        body = latexer() if callable(latexer) else None
        if isinstance(body, str):
            body = body.strip().strip("$").replace(r"\displaystyle", "").strip()
            display(Math(rf"{label} = {body}" if label is not None else body))
        else:
            if label is not None:
                print(f"{label}:")
            display(expr)
        return

    # headless: pretty text (SymPy's pretty printer when available)
    try:
        import sympy as sp

        text = sp.pretty(expr)
    except Exception:
        text = str(expr)
    if label is None:
        print(text)
    elif "\n" in text:
        print(f"{label} =\n{text}")
    else:
        print(f"{label} = {text}")
