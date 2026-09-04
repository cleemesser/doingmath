"""
Basis- and variance-aware matrices on top of SymPy.

SymPy's Matrix is a bare array of entries: it carries no record of which basis
its components refer to, or whether an index is contravariant or covariant.
This module wraps a sympy.Matrix in an object that carries that information and
refuses to contract indices that don't match.

Conventions
-----------
A slot is a pair (basis, variance), variance in {'up', 'down'}.

  'up'   = contravariant, components w.r.t. the basis vectors e_i
  'down' = covariant, components w.r.t. the dual basis e^i

Contraction is legal only between adjacent slots that share a basis and have
opposite variance. Everything else -- similarity vs. congruence in particular --
follows from that single rule.

A change of basis from f to e is stored as P with slots (e,'up') x (f,'down'),
i.e. the matrix whose columns are the f-basis vectors written in e-components,
so that v_e = P v_f.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp


@dataclass(frozen=True)
class Basis:
    """A named basis, optionally carrying the metric components in that basis.

    `metric` holds g_ij *as expressed in this basis*. Leaving it None means no
    inner product has been declared, so raising and lowering are unavailable and
    variance is strictly enforced.
    """

    name: str
    dim: int
    latex: Optional[str] = None  # defaults to \mathcal{<name>} when a bare letter
    metric: Optional[sp.ImmutableMatrix] = None

    def __post_init__(self):
        if self.metric is not None:
            g = sp.ImmutableMatrix(self.metric)
            if g.shape != (self.dim, self.dim):
                raise ValueError("metric shape does not match basis dimension")
            object.__setattr__(self, "metric", g)

    @staticmethod
    def euclidean(name: str, dim: int, **kw) -> "Basis":
        """An orthonormal basis of a positive-definite space: g = I."""
        return Basis(name, dim, metric=sp.eye(dim), **kw)

    @staticmethod
    def pseudo_orthonormal(name: str, signature, **kw) -> "Basis":
        """e.g. Basis.pseudo_orthonormal('m', [1, -1, -1, -1]) for Minkowski."""
        return Basis(name, len(signature), metric=sp.diag(*signature), **kw)

    @property
    def is_orthonormal(self) -> bool:
        """g is diagonal with entries +-1: the basis is orthonormal for its signature."""
        if self.metric is None:
            return False
        g = self.metric
        return g.is_diagonal() and all(
            sp.simplify(abs(g[i, i]) - 1) == 0 for i in range(self.dim)
        )

    @property
    def self_dual(self) -> bool:
        """g is exactly the identity, so raising and lowering are the identity map.

        This -- not orthonormality as such -- is what licenses ignoring variance.
        In a Minkowski frame the basis is orthonormal but raising an index flips
        signs, so variance still has to be tracked.
        """
        return (
            self.metric is not None
            and sp.simplify(sp.Matrix(self.metric) - sp.eye(self.dim)).is_zero_matrix
            is True
        )

    @property
    def tex(self) -> str:
        if self.latex is not None:
            return self.latex
        if len(self.name) == 1 and self.name.isalpha():
            return r"\mathcal{%s}" % self.name.upper()
        return r"\mathrm{%s}" % self.name

    def __repr__(self) -> str:
        mark = "*" if self.self_dual else ""
        return f"{self.name}{mark}({self.dim})"


@dataclass(frozen=True)
class Slot:
    basis: Basis
    variance: str  # 'up' or 'down'

    def __post_init__(self):
        if self.variance not in ("up", "down"):
            raise ValueError(f"variance must be 'up' or 'down', got {self.variance!r}")

    @property
    def dual(self) -> "Slot":
        return Slot(self.basis, "down" if self.variance == "up" else "up")

    def contracts_with(self, other: "Slot") -> bool:
        if self.basis != other.basis:
            return False
        if self.variance != other.variance:
            return True
        # Same variance: legal only when the metric is the identity in this
        # basis, in which case the contraction silently inserts delta_ij.
        return self.basis.self_dual

    def __repr__(self) -> str:
        if self.basis.self_dual:
            return f"{self.basis.name}~"
        arrow = "^" if self.variance == "up" else "_"
        return f"{self.basis.name}{arrow}"


class BasisMismatch(TypeError):
    pass


class TMatrix:
    """A sympy Matrix tagged with a row slot and a column slot.

    Either slot may be None, denoting a trivial (length-1) axis, so that
    column vectors, row covectors and scalars are representable.
    """

    def __init__(self, mat, row: Optional[Slot], col: Optional[Slot]):
        self.mat = sp.Matrix(mat)
        self.row = row
        self.col = col
        m, n = self.mat.shape
        if row is not None and row.basis.dim != m:
            raise ValueError(
                f"row slot {row} has dim {row.basis.dim}, matrix has {m} rows"
            )
        if row is None and m != 1:
            raise ValueError("trivial row slot requires a single row")
        if col is not None and col.basis.dim != n:
            raise ValueError(
                f"col slot {col} has dim {col.basis.dim}, matrix has {n} cols"
            )
        if col is None and n != 1:
            raise ValueError("trivial column slot requires a single column")

    # ---- construction helpers -------------------------------------------

    @staticmethod
    def vector(components, basis: Basis) -> "TMatrix":
        """A contravariant vector: components in the basis `basis`."""
        return TMatrix(sp.Matrix(components), Slot(basis, "up"), None)

    @staticmethod
    def covector(components, basis: Basis) -> "TMatrix":
        """A covariant vector (one-form) as a row."""
        return TMatrix(sp.Matrix([list(components)]), None, Slot(basis, "down"))

    @staticmethod
    def operator(mat, basis: Basis) -> "TMatrix":
        """A (1,1) tensor: a linear map of the space to itself."""
        return TMatrix(mat, Slot(basis, "up"), Slot(basis, "down"))

    @staticmethod
    def form(mat, basis: Basis) -> "TMatrix":
        """A (0,2) tensor: a bilinear form, e.g. a metric."""
        return TMatrix(mat, Slot(basis, "down"), Slot(basis, "down"))

    # ---- algebra ---------------------------------------------------------

    def __mul__(self, other):
        if isinstance(other, (int, float, sp.Expr)):
            return TMatrix(self.mat * other, self.row, self.col)
        if not isinstance(other, TMatrix):
            return NotImplemented
        if self.col is None or other.row is None:
            raise BasisMismatch("cannot contract over a trivial axis")
        if not self.col.contracts_with(other.row):
            raise BasisMismatch(
                f"cannot contract {self.col!r} with {other.row!r}: "
                "need the same basis and opposite variance"
            )
        return TMatrix(self.mat * other.mat, self.row, other.col)

    __rmul__ = lambda self, other: TMatrix(self.mat * other, self.row, self.col)

    def __add__(self, other):
        if not isinstance(other, TMatrix):
            return NotImplemented
        if (self.row, self.col) != (other.row, other.col):
            raise BasisMismatch(
                f"cannot add ({self.row!r},{self.col!r}) to ({other.row!r},{other.col!r})"
            )
        return TMatrix(self.mat + other.mat, self.row, self.col)

    @property
    def T(self) -> "TMatrix":
        return TMatrix(self.mat.T, self.col, self.row)

    def inv(self) -> "TMatrix":
        """Inverse map: slots swap position and flip variance."""
        if self.row is None or self.col is None:
            raise BasisMismatch("only square, non-trivial slots are invertible")
        return TMatrix(self.mat.inv(), self.col.dual, self.row.dual)

    # ---- metric operations ----------------------------------------------

    @staticmethod
    def metric_of(basis: Basis) -> "TMatrix":
        if basis.metric is None:
            raise BasisMismatch(f"no metric declared on basis {basis!r}")
        return TMatrix.form(basis.metric, basis)

    def lower(self, axis: str = "row") -> "TMatrix":
        """Contract the named axis with g, turning 'up' into 'down'."""
        slot = self.row if axis == "row" else self.col
        if slot is None or slot.variance == "down":
            raise BasisMismatch(f"{axis} slot {slot!r} is not contravariant")
        g = TMatrix.metric_of(slot.basis)
        return (g * self) if axis == "row" else (self * g)

    def raise_(self, axis: str = "row") -> "TMatrix":
        """Contract the named axis with g^-1, turning 'down' into 'up'."""
        slot = self.row if axis == "row" else self.col
        if slot is None or slot.variance == "up":
            raise BasisMismatch(f"{axis} slot {slot!r} is not covariant")
        ginv = TMatrix.metric_of(slot.basis).inv()
        return (ginv * self) if axis == "row" else (self * ginv)

    def adjoint(self) -> "TMatrix":
        """The metric adjoint g^-1 A^T g of a (1,1) tensor.

        Bare transposition of an operator is not basis-independent; it coincides
        with the adjoint exactly when g = I. This is the same fact as
        similarity = congruence on the orthogonal subgroup.
        """
        if self.row is None or self.col is None or self.row.basis != self.col.basis:
            raise BasisMismatch("adjoint is defined for endomorphisms")
        b = self.row.basis
        g = sp.Matrix(TMatrix.metric_of(b).mat)
        return TMatrix(g.inv() * self.mat.T * g, self.row, self.col)

    # ---- basis change ----------------------------------------------------

    def to(self, P: "TMatrix") -> "TMatrix":
        """Re-express in another basis using change-of-basis P.

        The correct transformation law is *deduced* from the slot variances:
        operators get a similarity, forms get a congruence, vectors get a
        single P. No branching on 'what kind of object is this' is needed.
        """
        Pi = P.inv()
        out = self
        # left slot
        if out.row is not None:
            left = P if out.row.contracts_with(P.col) else Pi
            if not left.col.contracts_with(out.row):
                left = P.T if P.T.col.contracts_with(out.row) else Pi.T
            out = left * out
        # right slot
        if out.col is not None:
            right = P if out.col.contracts_with(P.row) else Pi
            if not out.col.contracts_with(right.row):
                right = P.T if out.col.contracts_with(P.T.row) else Pi.T
            out = out * right
        return out

    # ---- display ---------------------------------------------------------
    #
    # The row slot decorates the opening bracket, the column slot the closing
    # bracket, and the *height* of each label carries the variance: raised for
    # contravariant, lowered for covariant. So an operator prints as
    #
    #        e            e                                 e
    #         [ ... ]         and a metric as  [ ... ]  ,  [ ... ]  is a vector.
    #                e                        e       e
    #
    # which is just index notation A^i_j, g_ij, v^i with the index names
    # replaced by the basis they are indices *over*.

    def _labels(self):
        """(text, latex) label for each slot, or None."""

        def one(slot):
            if slot is None:
                return None
            return slot.basis.name, slot.basis.tex, slot.variance, slot.basis.self_dual

        return one(self.row), one(self.col)

    def _block(self, body_lines) -> str:
        left, right = self._labels()
        # Self-dual basis: collapse to a single label, cartesian-tensor style.
        if left is not None and self.row.basis.self_dual:
            if right is None:
                right, left = left, None
            else:
                left = None
        h = len(body_lines)
        w = max(len(ln) for ln in body_lines)
        lines = [ln.ljust(w) for ln in body_lines]
        if left is not None:
            name, _, var, sd = left
            row = self._label_row(sd, var, h)
            lines = [
                (name if i == row else " " * len(name)) + ln
                for i, ln in enumerate(lines)
            ]
        if right is not None:
            name, _, var, sd = right
            row = self._label_row(sd, var, h)
            lines = [
                ln + (name if i == row else " " * len(name))
                for i, ln in enumerate(lines)
            ]
        return "\n".join(lines)

    @staticmethod
    def _label_row(self_dual, var, h) -> int:
        # A self-dual basis prints its label at mid-height: neither raised nor
        # lowered, because in that basis the distinction carries no information.
        if self_dual:
            return h // 2
        return 0 if var == "up" else h - 1

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm

        text = self._block(sp.pretty(self.mat).splitlines())
        return prettyForm(text, baseline=len(text.splitlines()) // 2)

    def _sympystr(self, printer, *args):
        return self._block(sp.pretty(self.mat).splitlines())

    def _latex(self, printer=None, *args):
        settings = dict(mat_delim="", mat_str="matrix")
        body = sp.latex(self.mat, **settings)
        left, right = self._labels()
        out = ""
        # For a self-dual (Euclidean orthonormal) basis, fall back to the usual
        # cartesian-tensor notation [A]_E: one subscript, no variance shown.
        if left is not None and not self.row.basis.self_dual:
            _, tex, var, _sd = left
            out += r"{}%s{%s}\!" % ("^" if var == "up" else "_", tex)
        out += r"\left[" + body + r"\right]"
        if right is not None:
            _, tex, var, sd = right
            if sd:
                out += r"_{%s}" % tex
            else:
                out += r"%s{%s}" % ("^" if var == "up" else "_", tex)
        elif left is not None and self.row.basis.self_dual:
            out += r"_{%s}" % left[1]
        return out

    def _repr_latex_(self) -> str:
        return r"$\displaystyle %s$" % self._latex()

    def __repr__(self) -> str:
        return self._sympystr(None)


def change_of_basis(P, source: Basis, target: Basis, check: bool = True) -> TMatrix:
    """Columns of P are the `source` basis vectors in `target` components.

    Then v_target = P @ v_source.

    If both bases declare a metric, P is checked against the compatibility
    condition g_source = P^T g_target P. For two Euclidean bases this reduces
    to P^T P = I, i.e. P must lie in O(n): declaring two bases orthonormal and
    then relating them by a non-isometry is the way the `self_dual` shortcut
    would otherwise start producing wrong answers.
    """
    P = sp.Matrix(P)
    if check and source.metric is not None and target.metric is not None:
        residual = sp.simplify(
            P.T * sp.Matrix(target.metric) * P - sp.Matrix(source.metric)
        )
        if not residual.is_zero_matrix:
            raise BasisMismatch(
                f"P is not an isometry {source!r} -> {target!r}: "
                f"P^T g_{target.name} P - g_{source.name} = {residual}"
            )
    return TMatrix(P, Slot(target, "up"), Slot(source, "down"))
