"""show_expr: native LaTeX in a notebook, pretty text when headless."""

import sympy as sp

from mathviz import backends, show_expr


def test_headless_pretty_prints_with_label(capsys):
    show_expr(sp.Matrix([[1, 2], [3, 4]]), label="A")
    out = capsys.readouterr().out
    assert "A =" in out and "1" in out and "4" in out


def test_headless_no_label(capsys):
    show_expr(sp.Rational(1, 3))
    assert "1/3" in capsys.readouterr().out


def test_notebook_renders_latex_math(monkeypatch):
    import IPython.display as ipd

    monkeypatch.setattr(backends, "in_notebook", lambda: True)
    monkeypatch.setattr(ipd, "Math", lambda s: ("MATH", s))
    captured = {}
    monkeypatch.setattr(ipd, "display", lambda obj: captured.setdefault("obj", obj))
    show_expr(sp.Matrix([[1, 2]]), label="M")
    kind, latex = captured["obj"]
    assert kind == "MATH" and latex.startswith("M = ")


def test_notebook_non_latex_object_falls_back_to_display(monkeypatch):
    import IPython.display as ipd

    monkeypatch.setattr(backends, "in_notebook", lambda: True)
    captured = {}
    monkeypatch.setattr(ipd, "display", lambda obj: captured.setdefault("obj", obj))
    show_expr([1, 2, 3])  # a plain list has no _repr_latex_
    assert captured["obj"] == [1, 2, 3]
