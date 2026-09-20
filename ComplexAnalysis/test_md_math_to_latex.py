"""Smoke test for md_math_to_latex.py — run: python3 test_md_math_to_latex.py"""
from md_math_to_latex import convert_line

CASES = [
    # (extraction artifact, expected markdown)
    ("the form _z_ = _x_ + i _y_ and", "the form $z = x + iy$ and"),
    ("if _f_ ( _z_ ) _̸_ = 0 then", "if $f(z)\\neq 0$ then"),
    ("_f_ : _D ⊂_ R[2] _→_ R[2] is", "$f : D \\subset \\mathbb{R}^{2}\\to \\mathbb{R}^{2}$ is"),
    ("that _f[′]_ ( _z_ ) = ~~_z_~~ _/|z|_[2] = 1 _/z_ ok",
     "that $f'(z) = \\bar{z}/|z|^{2} = 1/z$ ok"),
    ("for some _z_ 0, then _w_ 0 := _z_ 1", "for some $z_{0}$, then $w_{0} := z_{1}$"),
    ("Keep _prose emphasis_ alone", "Keep _prose emphasis_ alone"),
    ("C _\\ {_ 0 _}_ minus", "$\\mathbb{C} \\setminus \\lbrace 0 \\rbrace$ minus"),
    ("**Lemma 2.7.11** _Let γ be a path in D._",
     "**Lemma 2.7.11** _Let $\\gamma$ be a path in $D$._"),
    ("![](a/b.png) _z_ here", "![](a/b.png) $z$ here"),
]


def test():
    for src, want in CASES:
        got = convert_line(src)
        assert got == want, f"\n in: {src}\nget: {got}\nwant: {want}"
    print(f"ok — {len(CASES)} cases")


if __name__ == "__main__":
    test()
