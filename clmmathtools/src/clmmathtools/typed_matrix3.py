"""
Space-, frame- and variance-aware matrices on top of SymPy.

This is a redesign of :mod:`clmmathtools.typed_matrix`, splitting its single
``Basis`` type into two:

  ``VectorSpace``  the space itself -- dimension, and optionally a metric
  ``Frame``        one ordered basis of that space

The point of the split is the metric. In ``typed_matrix`` every ``Basis``
carried its own ``metric=`` matrix, so two bases of what you *meant* to be one
space were two independent declarations of geometry, reconciled only if you
happened to route them through ``change_of_basis(check=True)``. Here a
``VectorSpace`` holds g once, and a ``Frame``'s Gram matrix is *derived*:

    g_F = B^T g_ref B

with B the frame's vectors written in the space's reference frame. Two frames of
one space cannot disagree about the geometry, because only one of them states
it. For the same reason ``change_of_basis`` no longer takes a matrix -- it
computes P = B_target^-1 B_source -- so an inconsistent change of basis is not
merely rejected, it is unrepresentable.

Conventions
-----------
A slot is a pair (frame, variance), variance in {'up', 'down'}.

  'up'   = contravariant, components w.r.t. the frame vectors e_i
  'down' = covariant, components w.r.t. the dual frame e^i

Contraction is legal only between adjacent slots that share a frame and have
opposite variance. Everything else -- similarity vs. congruence in particular --
follows from that single rule.

A change of frame from f to e is stored as P with slots (e,'up') x (f,'down'),
i.e. the matrix whose columns are the f-frame vectors in e-components, so that
v_e = P v_f.

Storing g needs a frame: a matrix *is* components. So ``VectorSpace.metric``
holds g in the space's reference frame, which is the frame built by
``space.frame(name)`` with no columns given. That frame is privileged only in
being where the numbers were written down; nothing else in the module treats it
specially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp


class FrameMismatch(TypeError):
    pass


BasisMismatch = FrameMismatch  # the name typed_matrix used


@dataclass(frozen=True)
class VectorSpace:
    """A vector space, optionally equipped with a metric.

    `metric` holds g_ij in the space's *reference* frame. Leaving it None means
    no inner product has been declared, so raising and lowering are unavailable
    and variance is strictly enforced.
    """

    name: str
    dim: int
    metric: Optional[sp.ImmutableMatrix] = None
    latex: Optional[str] = None

    def __post_init__(self):
        if self.metric is None:
            return
        g = sp.ImmutableMatrix(self.metric)
        if g.shape != (self.dim, self.dim):
            raise ValueError("metric shape does not match the dimension of the space")
        if not sp.simplify(sp.Matrix(g) - sp.Matrix(g).T).is_zero_matrix:
            raise ValueError("a metric must be symmetric")
        object.__setattr__(self, "metric", g)

    # ---- construction helpers -------------------------------------------

    @staticmethod
    def euclidean(name: str, dim: int, **kw) -> "VectorSpace":
        """A positive-definite space whose reference frame is orthonormal."""
        return VectorSpace(name, dim, metric=sp.eye(dim), **kw)

    @staticmethod
    def pseudo_euclidean(name: str, signature, **kw) -> "VectorSpace":
        """e.g. VectorSpace.pseudo_euclidean('M', [1, -1, -1, -1]) for Minkowski."""
        return VectorSpace(name, len(signature), metric=sp.diag(*signature), **kw)

    def frame(
        self,
        name: str,
        cols=None,
        latex: Optional[str] = None,
        orthonormal: bool = False,
    ) -> "Frame":
        """A frame of this space; `cols` are its vectors in reference components.

        Omitting `cols` gives the reference frame itself. Passing
        `orthonormal=True` asserts that the resulting frame is orthonormal for
        this space's metric, and raises if it is not -- this is where a claim
        like "these two frames are both orthonormal" gets checked, rather than
        at change-of-frame time.
        """
        B = sp.eye(self.dim) if cols is None else sp.Matrix(cols)
        f = Frame(name, self, B, latex)
        if orthonormal and not f.is_orthonormal:
            raise FrameMismatch(
                f"frame {name!r} is not orthonormal for {self.name}: "
                f"B^T g B = {sp.simplify(sp.Matrix(f.metric))}"
            )
        return f

    # ---- properties ------------------------------------------------------

    @property
    def has_metric(self) -> bool:
        return self.metric is not None

    @property
    def is_degenerate(self) -> bool:
        """det g == 0: the flat map V -> V* is not invertible, so no raising.

        Not an error -- this is the projective / degenerate case, kingdon's
        Algebra(p, q, r) with r > 0 -- but it has to be caught before anything
        tries to invert g.
        """
        if self.metric is None:
            return False
        return sp.simplify(sp.Matrix(self.metric).det()) == 0

    @property
    def tex(self) -> str:
        if self.latex is not None:
            return self.latex
        return r"\mathbb{%s}" % self.name if len(self.name) == 1 else self.name

    def __repr__(self) -> str:
        if self.metric is None:
            return f"{self.name}({self.dim}, no metric)"
        return f"{self.name}({self.dim})"


@dataclass(frozen=True)
class Frame:
    """One ordered basis of a `VectorSpace`.

    `cols` holds the frame vectors as columns, in the space's reference-frame
    components. Everything metric about the frame is derived from that and the
    space's g -- a frame never declares a geometry of its own.
    """

    name: str
    space: VectorSpace
    cols: sp.ImmutableMatrix
    latex: Optional[str] = None

    def __post_init__(self):
        B = sp.ImmutableMatrix(self.cols)
        if B.shape != (self.space.dim, self.space.dim):
            raise ValueError(
                f"frame matrix is {B.shape}, expected "
                f"({self.space.dim}, {self.space.dim})"
            )
        if sp.simplify(sp.Matrix(B).det()) == 0:
            raise ValueError(f"frame {self.name!r} vectors are linearly dependent")
        object.__setattr__(self, "cols", B)

    @staticmethod
    def euclidean(name: str, dim: int) -> "Frame":
        """Shorthand: an orthonormal frame of a fresh anonymous Euclidean space.

        Convenient for one-off examples. Note that two frames minted this way
        belong to *different* spaces and so cannot be related -- when you want
        several frames of one space, build the `VectorSpace` first.
        """
        return VectorSpace.euclidean(f"V_{name}", dim).frame(name)

    # ---- derived geometry -------------------------------------------------

    @property
    def dim(self) -> int:
        return self.space.dim

    @property
    def metric(self) -> Optional[sp.ImmutableMatrix]:
        """g_ij in *this* frame: B^T g_ref B. Derived, never declared."""
        if self.space.metric is None:
            return None
        B = sp.Matrix(self.cols)
        return sp.ImmutableMatrix(sp.simplify(B.T * sp.Matrix(self.space.metric) * B))

    @property
    def is_orthonormal(self) -> bool:
        """g is diagonal with entries +-1: orthonormal for the space's signature."""
        g = self.metric
        if g is None:
            return False
        return g.is_diagonal() and all(
            sp.simplify(abs(g[i, i]) - 1) == 0 for i in range(self.dim)
        )

    @property
    def self_dual(self) -> bool:
        """g is exactly the identity in this frame, so raising/lowering is a no-op.

        This -- not orthonormality as such -- is what licenses ignoring variance.
        In a Minkowski frame the frame is orthonormal but raising an index flips
        signs, so variance still has to be tracked.
        """
        g = self.metric
        return (
            g is not None
            and sp.simplify(sp.Matrix(g) - sp.eye(self.dim)).is_zero_matrix is True
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
        return f"{self.name}{mark}({self.space.name}:{self.dim})"


@dataclass(frozen=True)
class Slot:
    frame: Frame
    variance: str  # 'up' or 'down'

    def __post_init__(self):
        if self.variance not in ("up", "down"):
            raise ValueError(f"variance must be 'up' or 'down', got {self.variance!r}")

    @property
    def space(self) -> VectorSpace:
        return self.frame.space

    @property
    def dual(self) -> "Slot":
        return Slot(self.frame, "down" if self.variance == "up" else "up")

    def contracts_with(self, other: "Slot") -> bool:
        if self.frame != other.frame:
            return False
        if self.variance != other.variance:
            return True
        # Same variance: legal only when the metric is the identity in this
        # frame, in which case the contraction silently inserts delta_ij.
        return self.frame.self_dual

    def __repr__(self) -> str:
        if self.frame.self_dual:
            return f"{self.frame.name}~"
        arrow = "^" if self.variance == "up" else "_"
        return f"{self.frame.name}{arrow}"


def _cob_matrix(source: Frame, target: Frame) -> sp.Matrix:
    """The plain sympy matrix P with v_target = P v_source."""
    if source.space != target.space:
        raise FrameMismatch(
            f"{source.name!r} and {target.name!r} are frames of different spaces "
            f"({source.space!r} vs {target.space!r})"
        )
    return sp.simplify(sp.Matrix(target.cols).inv() * sp.Matrix(source.cols))


def change_of_basis(source: Frame, target: Frame) -> "TMatrix":
    """P with slots (target,'up') x (source,'down'), so that v_target = P v_source.

    Unlike the `typed_matrix` version this takes no matrix and performs no
    compatibility check: P is computed from the two frames' own column matrices,
    so it is an isometry of the shared metric by construction. Declaring a frame
    to be something it is not is caught earlier, in `VectorSpace.frame`.
    """
    return TMatrix(
        _cob_matrix(source, target), Slot(target, "up"), Slot(source, "down")
    )


class TMatrix:
    """A sympy Matrix tagged with a row slot and a column slot.

    Either slot may be None, denoting a trivial (length-1) axis, so that column
    vectors, row covectors and scalars are representable. The two slots may live
    in different spaces, which is how a map f: V -> W is written.
    """

    def __init__(self, mat, row: Optional[Slot], col: Optional[Slot]):
        self.mat = sp.Matrix(mat)
        self.row = row
        self.col = col
        m, n = self.mat.shape
        if row is not None and row.frame.dim != m:
            raise ValueError(
                f"row slot {row} has dim {row.frame.dim}, matrix has {m} rows"
            )
        if row is None and m != 1:
            raise ValueError("trivial row slot requires a single row")
        if col is not None and col.frame.dim != n:
            raise ValueError(
                f"col slot {col} has dim {col.frame.dim}, matrix has {n} cols"
            )
        if col is None and n != 1:
            raise ValueError("trivial column slot requires a single column")

    # ---- construction helpers -------------------------------------------

    @staticmethod
    def vector(components, frame: Frame) -> "TMatrix":
        """A contravariant vector: components in the frame `frame`."""
        return TMatrix(sp.Matrix(components), Slot(frame, "up"), None)

    @staticmethod
    def covector(components, frame: Frame) -> "TMatrix":
        """A covariant vector (one-form) as a row."""
        return TMatrix(sp.Matrix([list(components)]), None, Slot(frame, "down"))

    @staticmethod
    def operator(mat, frame: Frame) -> "TMatrix":
        """A (1,1) tensor: a linear map of the space to itself."""
        return TMatrix(mat, Slot(frame, "up"), Slot(frame, "down"))

    @staticmethod
    def form(mat, frame: Frame) -> "TMatrix":
        """A (0,2) tensor: a bilinear form, e.g. a metric."""
        return TMatrix(mat, Slot(frame, "down"), Slot(frame, "down"))

    @staticmethod
    def map(mat, source: Frame, target: Frame) -> "TMatrix":
        """A linear map f: V -> W, components in `source` and `target` frames."""
        return TMatrix(mat, Slot(target, "up"), Slot(source, "down"))

    # ---- algebra ---------------------------------------------------------

    @staticmethod
    def _collapse_if_scalar(t: "TMatrix"):
        """Hand back a plain sympy scalar once an operation uses up every slot.

        A contraction with nothing left over *is* a number -- an inner product,
        a pairing, a trace -- so returning a 1x1 TMatrix would only make the
        caller unwrap it, and a 1x1 sympy Matrix is a poor stand-in for a number
        (``M == 5`` is silently False). The cost is that the return type of
        ``*`` is a union: TMatrix while slots remain, sympy Expr when they run
        out. That is the same bargain ``Matrix.dot`` makes.
        """
        return t.scalar if (t.row is None and t.col is None) else t

    def __mul__(self, other):
        if isinstance(other, (int, float, sp.Expr)):
            return self._collapse_if_scalar(
                TMatrix(self.mat * other, self.row, self.col)
            )
        if not isinstance(other, TMatrix):
            return NotImplemented
        if self.col is None or other.row is None:
            raise FrameMismatch("cannot contract over a trivial axis")
        if not self.col.contracts_with(other.row):
            raise FrameMismatch(self._why_not(self.col, other.row))
        return self._collapse_if_scalar(
            TMatrix(self.mat * other.mat, self.row, other.col)
        )

    @staticmethod
    def _why_not(a: Slot, b: Slot) -> str:
        """Explain a failed contraction, distinguishing the three ways it fails."""
        if a.space != b.space:
            return (
                f"cannot contract {a!r} with {b!r}: different spaces "
                f"({a.space!r} and {b.space!r})"
            )
        if a.frame != b.frame:
            return (
                f"cannot contract {a!r} with {b!r}: same space {a.space!r} but "
                f"different frames -- re-express one of them first, e.g. "
                f".in_frame({b.frame.name})"
            )
        if a.frame.space.metric is None:
            return (
                f"cannot contract {a!r} with {b!r}: same variance, and "
                f"{a.space!r} declares no metric to raise or lower with"
            )
        return (
            f"cannot contract {a!r} with {b!r}: same variance in a frame whose "
            f"metric is not the identity -- lower or raise one of them first"
        )

    def __rmul__(self, other):
        return self._collapse_if_scalar(TMatrix(self.mat * other, self.row, self.col))

    def __add__(self, other):
        if not isinstance(other, TMatrix):
            return NotImplemented
        if (self.row, self.col) != (other.row, other.col):
            raise FrameMismatch(
                f"cannot add ({self.row!r},{self.col!r}) to ({other.row!r},{other.col!r})"
            )
        return self._collapse_if_scalar(
            TMatrix(self.mat + other.mat, self.row, self.col)
        )

    @property
    def scalar(self):
        """The sympy scalar of a fully contracted result.

        Contraction already does this for you -- ``u.T * v`` hands back a sympy
        number, not a 1x1 TMatrix -- so this is the explicit route, for a
        zero-slot TMatrix built directly, and the guard is the useful part: a
        leftover slot means the answer is a vector or an operator, not a number.

        The unwrapping matters because a 1x1 sympy Matrix is emphatically *not*
        a number. ``M + 1`` and ``float(M)`` raise, and -- the trap -- ``M == 5``
        returns False rather than erroring, so a test against a 1x1 silently
        always fails.
        """
        if self.row is not None or self.col is not None:
            raise FrameMismatch(
                f"not a scalar: slots ({self.row!r}, {self.col!r}) remain -- "
                f"this is a {self.mat.shape[0]}x{self.mat.shape[1]} array of components"
            )
        return self.mat[0, 0]

    @property
    def T(self) -> "TMatrix":
        return TMatrix(self.mat.T, self.col, self.row)

    def inv(self) -> "TMatrix":
        """Inverse map: slots swap position and flip variance."""
        if self.row is None or self.col is None:
            raise FrameMismatch("only square, non-trivial slots are invertible")
        return TMatrix(self.mat.inv(), self.col.dual, self.row.dual)

    # ---- metric operations ----------------------------------------------

    @staticmethod
    def metric_of(frame: Frame) -> "TMatrix":
        if frame.space.metric is None:
            raise FrameMismatch(f"no metric declared on space {frame.space!r}")
        return TMatrix.form(frame.metric, frame)

    @staticmethod
    def _inverse_metric(frame: Frame) -> sp.Matrix:
        if frame.space.metric is None:
            raise FrameMismatch(f"no metric declared on space {frame.space!r}")
        if frame.space.is_degenerate:
            raise FrameMismatch(
                f"the metric on {frame.space!r} is degenerate (det g = 0), so the "
                f"flat map V -> V* has no inverse: indices cannot be raised"
            )
        return sp.Matrix(frame.metric).inv()

    def _resolve_slot(self, slot) -> str:
        """Map a slot index to the axis ('row' or 'col') it names.

        Slot indices count the object's *real* slots, skipping trivial
        (length-1) axes. So a covector stored 1xn has exactly one slot, `slot=0`,
        even though it occupies array position 1 -- the index is a statement
        about the tensor, not about how it is laid out.

        A vector or covector therefore needs no argument at all: there is only
        one slot to act on. An operator or a bilinear form has two, and must say
        which, rather than have a default quietly pick one.
        """
        present = [
            name
            for name, sl in (("row", self.row), ("col", self.col))
            if sl is not None
        ]
        if not present:
            raise FrameMismatch("a scalar has no slots to raise or lower")
        if slot is None:
            if len(present) == 1:
                return present[0]
            raise FrameMismatch(
                "this object has two slots, so which one to act on is ambiguous: "
                "pass slot=0 for the first (row) or slot=1 for the second (column)"
            )
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise ValueError(
                f"slot must be 0 or 1, or omitted for a one-slot object, got {slot!r}"
            )
        if not 0 <= slot < len(present):
            raise ValueError(
                f"slot {slot} is out of range: this object has {len(present)} "
                f"slot{'s' if len(present) > 1 else ''}"
            )
        return present[slot]

    def lower(self, slot: int | None = None) -> "TMatrix":
        """Contract a slot with g, turning 'up' into 'down'.

        `slot` is 0 for the first slot and 1 for the second, counting only
        non-trivial slots. Omit it on a vector or covector, which has just one;
        it is required on an operator or a form, which have two.
        """
        axis = self._resolve_slot(slot)
        sl = self.row if axis == "row" else self.col
        if sl.variance == "down":
            raise FrameMismatch(f"{axis} slot {sl!r} is already covariant")
        g = TMatrix.metric_of(sl.frame)
        return (g * self) if axis == "row" else (self * g)

    def raise_(self, slot: int | None = None) -> "TMatrix":
        """Contract a slot with g^-1, turning 'down' into 'up'.

        `slot` is 0 for the first slot and 1 for the second, counting only
        non-trivial slots. Omit it on a vector or covector, which has just one;
        it is required on an operator or a form, which have two.
        """
        axis = self._resolve_slot(slot)
        sl = self.row if axis == "row" else self.col
        if sl.variance == "up":
            raise FrameMismatch(f"{axis} slot {sl!r} is already contravariant")
        ginv = TMatrix(
            TMatrix._inverse_metric(sl.frame),
            Slot(sl.frame, "up"),
            Slot(sl.frame, "up"),
        )
        return (ginv * self) if axis == "row" else (self * ginv)

    @property
    def variance(self) -> tuple:
        """The variance of each real slot, in order: e.g. ('up', 'down')."""
        return tuple(sl.variance for sl in (self.row, self.col) if sl is not None)

    def with_variance(self, *variances) -> "TMatrix":
        """Re-express with the requested variance on each slot.

        Give one variance per real slot -- 'up'/'down', or OGRePy-style +1/-1 --
        either as separate arguments or as a single tuple::

            A.with_variance("down", "down")     # lower whichever slots are up
            A.with_variance((-1, -1))           # the same thing
            v.with_variance("down")             # a vector: one slot

        This states the *destination* rather than the operation, so nothing has
        to name a slot: going from ('up', 'down') to ('down', 'down') already
        says "lower the first index". `lower` and `raise_` remain the explicit
        form, for when the point is the step rather than the result. Slots
        already in the requested variance are left untouched, so this is a no-op
        when there is nothing to do.
        """
        if len(variances) == 1 and isinstance(variances[0], (list, tuple)):
            variances = tuple(variances[0])
        present = [
            name
            for name, sl in (("row", self.row), ("col", self.col))
            if sl is not None
        ]
        want = [self._variance_name(v) for v in variances]
        if len(want) != len(present):
            n = len(present)
            raise ValueError(
                f"expected {n} variance{'s' if n != 1 else ''} for a {n}-slot "
                f"object, got {len(want)}"
            )
        out = self
        for i, target in enumerate(want):
            sl = out.row if present[i] == "row" else out.col
            if sl.variance != target:
                out = out.lower(i) if target == "down" else out.raise_(i)
        return out

    @staticmethod
    def _variance_name(v) -> str:
        """Normalise a variance specifier to 'up' or 'down'."""
        if isinstance(v, bool):
            raise ValueError(f"variance must be 'up'/'down' or +1/-1, got {v!r}")
        if v in ("up", "down"):
            return v
        if v == 1:
            return "up"
        if v == -1:
            return "down"
        raise ValueError(f"variance must be 'up'/'down' or +1/-1, got {v!r}")

    def adjoint(self) -> "TMatrix":
        """The metric adjoint of a linear map, g_V^-1 A^T g_W.

        For f: V -> W with slots (W,'up') x (V,'down'), the adjoint is the
        unique f†: W -> V with

            g_V(f† w, u) = g_W(w, f u)   for all u in V, w in W

        and it comes back with slots (V,'up') x (W,'down'). Unlike
        `typed_matrix.adjoint` this is not restricted to endomorphisms -- the
        two ends may be different spaces, which is the coordinate-free
        transpose (the pullback f*: W* -> V*) with both ends carried back
        through their own metrics. When V is W and g = I it degenerates to the
        bare transpose.
        """
        if self.row is None or self.col is None:
            raise FrameMismatch("adjoint is defined for maps with two slots")
        target, source = self.row.frame, self.col.frame  # f: source -> target
        gW = sp.Matrix(TMatrix.metric_of(target).mat)
        gV_inv = TMatrix._inverse_metric(source)
        return TMatrix(
            sp.simplify(gV_inv * self.mat.T * gW),
            Slot(source, "up"),
            Slot(target, "down"),
        )

    # ---- change of frame -------------------------------------------------

    def in_frame(self, target: Frame) -> "TMatrix":
        """Re-express every slot in `target`, deducing each transformation law.

        The law is read straight off the variance, one slot at a time:

            row  'up' -> P        row  'down' -> P^-T
            col  'up' -> P^T      col  'down' -> P^-1

        so an operator (up, down) picks up a similarity P A P^-1, a form
        (down, down) a congruence P^-T g P^-1, and a vector a single P -- with
        no branching on "what kind of object is this". Slots already in
        `target`, or in another space entirely, are left alone.
        """
        mat, row, col = self.mat, self.row, self.col
        if row is not None and row.frame != target and row.space == target.space:
            P = _cob_matrix(row.frame, target)
            mat = (P if row.variance == "up" else P.inv().T) * mat
            row = Slot(target, row.variance)
        if col is not None and col.frame != target and col.space == target.space:
            P = _cob_matrix(col.frame, target)
            mat = mat * (P.T if col.variance == "up" else P.inv())
            col = Slot(target, col.variance)
        return TMatrix(sp.simplify(mat), row, col)

    def to(self, P: "TMatrix") -> "TMatrix":
        """Re-express using an explicit change-of-frame P (see `in_frame`).

        Kept from `typed_matrix`: rather than reading the law off the variance,
        this searches P, P^-1, P^T, P^-T for the one whose slot contracts. Same
        answer, and it assumes both slots start in the same frame.
        """
        Pi = P.inv()
        out = self
        if out.row is not None:
            left = P if out.row.contracts_with(P.col) else Pi
            if not left.col.contracts_with(out.row):
                left = P.T if P.T.col.contracts_with(out.row) else Pi.T
            out = left * out
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
    # replaced by the frame they are indices *over*.

    def _labels(self):
        """(name, latex, variance, self_dual) for each slot, or None."""

        def one(slot):
            if slot is None:
                return None
            return slot.frame.name, slot.frame.tex, slot.variance, slot.frame.self_dual

        return one(self.row), one(self.col)

    @property
    def _collapse(self) -> bool:
        """Print one label instead of two, cartesian-tensor style.

        Only legal when *both* slots live in the same self-dual frame: then
        variance carries no information and [A]_E says everything. A mixed
        object like a change of frame (row in e, column in f) must keep both
        labels -- collapsing it would print P as though it were a matrix in the
        f frame alone.
        """
        if self.row is None or not self.row.frame.self_dual:
            return False
        return self.col is None or self.col.frame == self.row.frame

    def _block(self, body_lines) -> str:
        left, right = self._labels()
        if left is not None and self._collapse:
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
        # A self-dual frame prints its label at mid-height: neither raised nor
        # lowered, because in that frame the distinction carries no information.
        if self_dual:
            return h // 2
        return 0 if var == "up" else h - 1

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm

        text = self._block(sp.pretty(self.mat).splitlines())
        return prettyForm(text, baseline=len(text.splitlines()) // 2)

    def _sympystr(self, printer, *args):
        return self._block(sp.pretty(self.mat).splitlines())

    @staticmethod
    def _label_tex(tex: str, var: str, self_dual: bool, side: str) -> str:
        """A slot label placed beside the bracket, in LaTeX.

        The *height* carries the variance, exactly as in the text rendering:
        raised for contravariant, lowered for covariant, and mid-height for a
        self-dual frame, where the distinction carries no information.

        Mid-height comes for free: a ``\\left[...\\right]`` group is centred on
        the math axis, so an unshifted label beside it already sits level with
        the bracket's middle. Setting it in ``\\scriptstyle`` keeps it the size
        of the sub- and superscripts it replaces, so it reads as an annotation
        rather than as a factor multiplying the matrix.
        """
        if self_dual:
            label = r"{\scriptstyle %s}" % tex
            return label if side == "right" else label + r"\,"
        script = "^" if var == "up" else "_"
        if side == "left":
            return r"{}%s{%s}\!" % (script, tex)
        return r"%s{%s}" % (script, tex)

    def _latex(self, printer=None, *args):
        settings = dict(mat_delim="", mat_str="matrix")
        body = sp.latex(self.mat, **settings)
        left, right = self._labels()
        out = ""
        if left is not None and not self._collapse:
            _, tex, var, sd = left
            out += self._label_tex(tex, var, sd, side="left")
        out += r"\left[" + body + r"\right]"
        if right is not None:
            _, tex, var, sd = right
            out += self._label_tex(tex, var, sd, side="right")
        elif left is not None and self._collapse:
            _, tex, var, sd = left
            out += self._label_tex(tex, var, sd, side="right")
        return out

    def _repr_latex_(self) -> str:
        return r"$\displaystyle %s$" % self._latex()

    def __repr__(self) -> str:
        return self._sympystr(None)
