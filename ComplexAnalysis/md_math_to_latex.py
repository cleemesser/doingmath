#!/usr/bin/env python3
"""Convert PDF-extraction math artifacts in markdown to inline LaTeX.

The extractor wrote inline math as italic runs (`_z_`), `[2]` for super/sub-
scripts, `~~z~~` for conjugates and raw unicode math glyphs; whole theorem
statements are one italic run with math sprinkled inside.  Display equations
are PNG images in this dump, so only inline math needs converting.

Usage: python3 md_math_to_latex.py IN.md [OUT.md]
"""
import re
import sys

SYMBOLS = {
    "−": "-", "…": "\\ldots", "⋯": "\\cdots",
    "±": "\\pm", "∓": "\\mp", "×": "\\times", "·": "\\cdot", "÷": "\\div",
    "≤": "\\le", "≥": "\\ge", "≠": "\\neq", "≈": "\\approx", "≡": "\\equiv",
    "∼": "\\sim", "≅": "\\cong", "∝": "\\propto",
    "⊂": "\\subset", "⊃": "\\supset", "⊆": "\\subseteq", "⊇": "\\supseteq",
    "∈": "\\in", "∉": "\\notin", "∪": "\\cup", "∩": "\\cap",
    "∀": "\\forall", "∃": "\\exists", "∅": "\\emptyset", "∞": "\\infty",
    "∂": "\\partial", "∑": "\\sum", "∏": "\\prod", "∫": "\\int", "√": "\\sqrt",
    "°": "^{\\circ}", "′": "'", "″": "''", "∗": "*",
    "◦": "\\circ", "→": "\\to", "←": "\\leftarrow", "↔": "\\leftrightarrow",
    "↦": "\\mapsto", "⟶": "\\longrightarrow", "⇒": "\\Rightarrow",
    "⇔": "\\Leftrightarrow", "↑": "\\uparrow", "↓": "\\downarrow",
    "↖": "\\nwarrow", "↗": "\\nearrow", "↙": "\\swarrow", "↘": "\\searrow",
    "⊕": "\\oplus", "⊖": "\\ominus", "⊗": "\\otimes", "△": "\\triangle",
    "∠": "\\angle", "∥": "\\parallel", "⌈": "\\lceil", "⌉": "\\rceil",
    "⌊": "\\lfloor", "⌋": "\\rfloor", "⋅": "\\cdot", "∖": "\\setminus",
    "∧": "\\wedge", "∨": "\\vee", "≺": "\\prec", "∘": "\\circ",
}
GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "varepsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "ς": "sigma", "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi",
    "Π": "Pi", "Σ": "Sigma", "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi",
    "Ω": "Omega", "ϑ": "vartheta", "ϱ": "varrho", "ϵ": "epsilon",
}
FUNCS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth",
    "arcsin", "arccos", "arctan", "log", "ln", "lg", "exp", "arg", "Re", "Im",
    "det", "dim", "deg", "mod", "gcd", "lcm", "max", "min", "sup", "inf",
    "lim", "Res", "sign", "div", "rot", "grad", "ord", "dist", "diag", "Tr",
}
MATH_GLYPHS = set(SYMBOLS) | set(GREEK) | {"\u0338", "\u02d9"}
# two-letter italic runs that are English, not `x_y` ( _fn_ , _zk_ , _Dk_ ... )
STOPWORDS2 = {"in", "is", "of", "as", "at", "it", "be", "by", "to", "we", "he",
              "or", "no", "so", "if", "us", "me", "my", "on", "do", "up"}
INDEXED = "nkmjilswtrpq"
UMLAUT = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}
BBOLD = "RCHNZQ"
NEUTRAL_CHARS = set("0123456789()[]|.,:;+-*/<>=~^_!?\\'\"{}")
PROSE_WORD = re.compile(r"[A-Za-z\u00c0-\u00ff'\u2019\-\u2013\u2014]{2,}$")
FUNC_RE = re.compile(r"(?<![A-Za-z\\])(" +
                     "|".join(sorted(FUNCS, key=len, reverse=True)) +
                     r")(?![A-Za-z])")
# upright number-set letters in the source (not inside an italic run)
BBOLD_SRC = re.compile(r"(?<![A-Za-z0-9\\. ])([" + BBOLD + r"])(?![A-Za-z0-9.])")
BBOLD_MARK = re.compile("\x06([" + BBOLD + "])")
ITALIC_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZΑ-ϕ")
SUP_MERGE = re.compile(r"\^\{([^{}]*)\}\s*\^\{([^{}]*)\}")
WS_RE = re.compile(r"\s+")
ITALIC_RE = re.compile(r"_(?!\s|_)(.+?)(?<!\s)_(?![A-Za-z0-9(\\])")
STRAY_RE = re.compile(r"(?<![A-Za-z\\])((?:[" +
                      re.escape("".join(GREEK) + "".join(SYMBOLS)) + r"])+)")

MO, MC, EO, EC = "\x01", "\x02", "\x03", "\x04"   # math atom / em atom


def esc_source(s: str) -> str:
    """LaTeX-ify raw source text: setminus and literal braces."""
    s = re.sub(r"\\{1,2}", lambda m: "\\setminus ", s)
    return s.replace("{", "\\lbrace ").replace("}", " \\rbrace")


class Ctx:
    def __init__(self):
        self.math = []
        self.em = []

    def add_math(self, body):
        if (re.fullmatch(r"[A-Za-z\u0391-\u03d5][" + INDEXED + "]", body)
                and body not in STOPWORDS2):
            body = body[0] + "_{" + body[1] + "}"      # _fn_ -> f_n
        self.math.append(body)
        return f"{MO}{len(self.math) - 1}{MC}"

    def add_em(self, body):
        self.em.append(body)
        return f"{EO}{len(self.em) - 1}{MC}"


# --------------------------------------------------------------- classification
def is_math_italic(body: str) -> bool:
    """Is the italic run `_body_` pure math (True) or prose (False)?"""
    raw = body.strip()
    b = raw.rstrip(".,;:!?") if re.search(r"[A-Za-z]", raw) else raw
    if not b or "$" in b:
        return False
    if b in GREEK or b in FUNCS:
        return True
    if re.fullmatch(r"[A-Za-z\u0391-\u03d5]\d", b):
        return True
    if re.fullmatch(r"[A-Za-z\u0391-\u03d5][a-z]", b):
        return b not in STOPWORDS2
    if not re.search(r"[A-Za-z\u0391-\u03d5]", b):      # _→_ _/z_ _̸_
        return True
    words = re.findall(r"[A-Za-z]{2,}", b)
    if words and not all(w in FUNCS for w in words):
        return False
    return (any(ch in MATH_GLYPHS for ch in b)
            or bool(re.search(r"[=<>+\-/*|^\[\](),]", b))
            or not words)


def is_neutral(tok: str) -> bool:
    """Glue token that may sit inside a math island."""
    if not tok:
        return False
    if tok in FUNCS or re.fullmatch(r"[i en]", tok):
        return True
    if re.fullmatch(r"[" + BBOLD + r"]{1,2}", tok):
        return True
    if re.fullmatch(r"[A-Za-z\u0391-\u03d5]\[[^\[\]]{1,6}\]", tok):   # R[2]
        return True
    return all(ch in NEUTRAL_CHARS for ch in tok)


def is_mathy_token(t: str) -> bool:
    """A word inside a prose run that is really math (`γ`, `D`, `z[−]`)."""
    t = t.strip(".,;:!?")
    if not t or t in ("a", "A", "I"):
        return False
    if t in FUNCS:
        return True
    if any(ch in GREEK for ch in t):
        return not re.search(r"[A-Za-z]{2,}", t)
    if re.fullmatch(r"[A-Za-z]\d?(\[[^\[\]]{1,8}\])?['′″]?", t):
        return True
    if re.fullmatch(r"[A-Za-zΑ-ϕ][" + INDEXED + "]", t):
        return t not in STOPWORDS2                                   # fn -> f_n
    if re.fullmatch(r"[0-9]{1,2}", t):
        return True
    if re.search(r"[A-Za-zΑ-ϕ]", t):
        return False
    return all(ch in MATH_GLYPHS or ch in NEUTRAL_CHARS for ch in t)


# --------------------------------------------------------------------- latex
def to_latex(s: str) -> str:
    s = re.sub(r"\[\s*['\u2032\u2033]{1,2}\s*\]",
               lambda m: m.group(0)[1:-1], s)            # f['] -> f'
    s = re.sub(r"~~(.+?)~~", lambda m: "\\bar{" + m.group(1) + "}", s)
    # attached brackets are exponents: R[2], e[-], z[n]  (not intervals)
    s = re.sub(r"([A-Za-z0-9)\|Α-ϕ′'])\s?\[(?![\s,])([^\[\],\s\n]{1,12}?)\]",
               lambda m: (m.group(1) + "^{" + m.group(2) + "}")
               if "\ufffd" not in m.group(2) else m.group(0), s)
    s = s.replace("[˙]", "^{\\cdot}")
    s = re.sub(r"\s*\u0338\s*=", lambda m: " \\neq ", s)
    s = re.sub(r"\u0338", lambda m: "\\neq ", s)
    for ch, name in GREEK.items():
        s = s.replace(ch, "\\" + name + " ")
    for ch, rep in SYMBOLS.items():
        s = s.replace(ch, " " + rep + " ")
    s = s.replace("\u02d9", "^{\\cdot}")
    s = FUNC_RE.sub(lambda m: "\\" + m.group(1), s)
    s = s.replace("\\Re", "\\mathrm{Re}").replace("\\Im", "\\mathrm{Im}")
    while SUP_MERGE.search(s):
        s = SUP_MERGE.sub(lambda m: "^{" + m.group(1) + m.group(2) + "}", s)
    # a bare `\` in this dump means setminus:  D \ J
    s = re.sub(r"\\(?![A-Za-z{},\\])", lambda m: "\\setminus ", s)
    return s


def clean_spacing(s: str) -> str:
    s = WS_RE.sub(" ", s).strip()
    s = re.sub(r"\s*([)\]}])\s*", lambda m: m.group(1), s)
    s = re.sub(r"([(\[])\s*", lambda m: m.group(1), s)
    s = re.sub(r"\s*/\s*", lambda m: "/", s)
    s = re.sub(r"\s*([_^])\s*", lambda m: m.group(1), s)
    s = re.sub(r"\s*([,;])\s*", lambda m: m.group(1) + " ", s)
    s = re.sub(r"\s*(['′″])\s*", lambda m: m.group(1), s)
    # function application:  f (z) -> f(z)
    s = re.sub(r"(?<![A-Za-z\\])([A-Za-z])\s+\(", lambda m: m.group(1) + "(", s)
    s = re.sub(r"([)\]\|}'′″])\s+\(", lambda m: m.group(1) + "(", s)
    # imaginary unit:  i v -> iv
    s = re.sub(r"(?<=[\s+\-(,=])i\s+(?=[a-zA-Z]{1,2}(?![a-zA-Z]))",
               lambda m: "i", s)
    s = re.sub(r"\s*:=\s*", lambda m: " := ", s)
    s = re.sub(r"(?<=[^\s:\\(<])=(?![=>])", lambda m: " = ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return re.sub(r"^[\s,;]+|[,;]+$", "", s).strip()


def render_math_tokens(tokens, bbold=True) -> str:
    """Render a run of math tokens as one `$...$` span."""
    out, prev = [], ""
    for t in tokens:                                    # z 0 -> z_{0}
        m = re.fullmatch(r"(\d{1,2})([.,;:)\]]*)", t)
        word = re.search(r"[A-Za-z]+$", prev)
        ok = bool(prev) and (prev[-1] in "})'′" or
                             (prev[-1].isalpha()
                              and not (word and word.group(0) in FUNCS)
                              and not re.search(r"\\[A-Za-z]+$", prev)))
        out.append("_{%s}%s" % (m.group(1), m.group(2)) if m and ok else t)
        prev = t
    body = clean_spacing(to_latex(" ".join(out)))
    body = BBOLD_MARK.sub(lambda m: "\\mathbb{%s}" % m.group(1), body)
    if not body or "$" in body:
        return ""
    trail = ""
    m = re.search(r"([.,])$", body)
    if m:
        trail, body = m.group(1), body[:-1].rstrip()
    return f"${body}${trail}"


def inner_math(body: str) -> str:
    """Prose run with math words inside: `_Let γ be a path in D ⊂_`."""
    toks = [t for t in body.split(" ") if t]
    flags = [is_mathy_token(t) for t in toks]
    for i, t in enumerate(toks):        # glue neutral tokens between math words
        if (not flags[i] and t and all(ch in NEUTRAL_CHARS for ch in t)
                and i and flags[i - 1] and i + 1 < len(toks) and flags[i + 1]):
            flags[i] = True
    out, i = [], 0
    while i < len(toks):
        if not flags[i]:
            out.append(toks[i])
            i += 1
            continue
        grp = []
        while i < len(toks) and flags[i]:
            grp.append(toks[i])
            i += 1
        if any(re.search(r"[A-Za-z\u0391-\u03d5]", g) for g in grp):
            out.append(render_math_tokens([esc_source(g) for g in grp]))
        else:
            out.extend(grp)             # bare numbers stay prose
    return " ".join(out)


# ---------------------------------------------------------------- line driver
def process_segment(text: str, ctx: "Ctx") -> str:
    if "_" not in text and "~" not in text:
        return text

    def italic_repl(m):
        body = m.group(1)
        if is_math_italic(body):
            return ctx.add_math(esc_source(body))
        if any(is_mathy_token(t) for t in body.split(" ")):
            return ctx.add_em(inner_math(body))
        return ctx.add_em(body)

    text = ITALIC_RE.sub(italic_repl, text)

    def conj(m):
        inner = re.sub(MO + r"(\d+)" + MC,
                       lambda k: ctx.math[int(k.group(1))], m.group(1))
        return ctx.add_math("\\bar{" + inner + "}")

    text = re.sub(r"~~(" + MO + r"\d+" + MC + r")~~", conj, text)
    text = re.sub(r"~~([A-Za-z\u0391-\u03d5])~~",
                  lambda m: "\\bar{" + m.group(1) + "}", text)

    kinds = []
    for p in [p for p in re.split(r"(\s+)", text) if p != ""]:
        if p.isspace():
            kinds.append(("ws", p))
        elif MO in p or EO in p:
            kinds.append(("math" if MO in p else "word", p))
        elif is_neutral(p):
            kinds.append(("neutral", p))
        else:
            kinds.append(("word", p))

    keep = [k[0] == "math" for k in kinds]
    anchors = [i for i, (k, _) in enumerate(kinds) if k == "math"]
    for a, b in zip(anchors, anchors[1:]):   # glue neutrals between two atoms
        if all(kinds[j][0] in ("neutral", "ws") for j in range(a + 1, b)):
            for j in range(a, b + 1):
                keep[j] = True

    def neighbour(i, step):
        j = i + step
        while 0 <= j < len(kinds) and kinds[j][0] == "ws":
            j += step
        return (j, kinds[j]) if 0 <= j < len(kinds) else (-1, ("edge", ""))

    def mathy(k, t):
        return k == "math" or MO in t or bool(re.fullmatch(
            r"\d{1,2}|[A-Za-z](\[[^\]]{1,6}\])?", t.strip(".,;:")))

    # extend islands over math-looking neighbours:  z 0 , R[2] , = 0
    changed = True
    while changed:
        changed = False
        for i, (k, t) in enumerate(kinds):
            if keep[i] or k != "neutral":
                continue
            li, (lk, lt) = neighbour(i, -1)
            ri, (rk, rt) = neighbour(i, 1)
            if not ((li >= 0 and keep[li]) or (ri >= 0 and keep[ri])):
                continue
            operand = (re.fullmatch(r"\d{1,2}[.,;:)\]]?", t)
                       or re.fullmatch(r"[A-Za-z](\[[^\]]{1,6}\])?", t)
                       or re.fullmatch(r"\[[^\[\],\s]{1,12}\]", t))
            operator = re.fullmatch(r"[:=<>+\-*/]+", t)
            elif_ok = False
            if (re.fullmatch(r"[)\]}]{1,2}[.,;:]*", t) and li >= 0
                    and keep[li]):
                # closing bracket: take it when the island has an open one
                j, bal = li, 0
                while j >= 0 and keep[j]:
                    txt = re.sub(MO + r"\d+" + MC, "", kinds[j][1])
                    bal += sum(c in "([{" for c in txt)
                    bal -= sum(c in ")]}" for c in txt)
                    j -= 1
                elif_ok = bal > 0
            if operand or (operator and ri >= 0 and mathy(rk, rt)) or elif_ok:
                keep[i] = True
                changed = True

    def plain(toks):
        # mark upright number-set letters (only outside math atoms)
        def one(t):
            if MO in t:
                return re.sub(MO + r"\d+" + MC,
                              lambda m: ctx.math[int(m.group(0)[1:-1])], t)
            return BBOLD_SRC.sub(lambda m: "\x06" + m.group(1), t)
        return " ".join(one(t) for t in toks).split(" ")

    res, i = [], 0
    while i < len(kinds):
        if not keep[i]:
            res.append(kinds[i][1])
            i += 1
            continue
        island = []
        while i < len(kinds):
            k, t = kinds[i]
            if keep[i]:
                island.append(t)
                i += 1
            elif k == "ws" and island and i + 1 < len(kinds) and keep[i + 1]:
                island.append(" ")
                i += 1
            else:
                break
        while island and island[-1] == " ":
            island.pop()
        joined = [t for t in plain(island) if t]
        res.append(render_math_tokens(joined))
    return "".join(res)


def wrap_stray(line: str) -> str:
    """Greek/math glyphs left outside math spans (upright in the PDF)."""
    line = re.sub(r"([aouAOU])\u00a8", lambda m: UMLAUT[m.group(1)], line)

    def repl(m):
        body = " ".join("\\" + GREEK[c] if c in GREEK else SYMBOLS[c]
                        for c in m.group(1))
        return f"${body}$"

    def sets(m):
        return f"$\\mathbb{{{m.group(1)}}}$"

    parts = re.split(r"(\$[^$]*\$)", line)
    for i in range(0, len(parts), 2):
        parts[i] = STRAY_RE.sub(repl, parts[i])
        parts[i] = BBOLD_SRC.sub(sets, parts[i])
    return "".join(parts)


def convert_line(line: str) -> str:
    if line.startswith("|") or "<br>" in line or not line.strip():
        return line
    ctx = Ctx()
    out, pos = [], 0
    for m in re.finditer(r"!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|`[^`]*`",
                         line):
        out.append(process_segment(line[pos:m.start()], ctx))
        out.append(m.group(0))
        pos = m.end()
    out.append(process_segment(line[pos:], ctx))
    s = "".join(out)
    s = re.sub(EO + r"(\d+)" + MC, lambda m: f"_{ctx.em[int(m.group(1))]}_", s)
    s = re.sub(MO + r"(\d+)" + MC, lambda m: ctx.math[int(m.group(1))], s)
    # punctuation and hyphens hug the preceding formula:  `$n$ -th` -> `$n$-th`
    s = re.sub(r"\$ (?=[.,;:\-])", lambda m: "$", s)
    return wrap_stray(s)


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    lines = open(src, encoding="utf-8").read().split("\n")
    open(dst, "w", encoding="utf-8").write(
        "\n".join(convert_line(l) for l in lines))
    print(f"{src} -> {dst}: {len(lines)} lines")


if __name__ == "__main__":
    main()
