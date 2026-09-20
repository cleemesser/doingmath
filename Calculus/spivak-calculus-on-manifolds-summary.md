# Calculus on Manifolds (Michael Spivak, 1965) — symbols and concepts by chapter

A summary of `spivak-calculus-on-manifolds.pdf` (146 pp.).
Companion file: `spivak-calculus-on-manifolds-summary.tex`.

> **Note on the math in tables.** Vertical bars are written `\lvert`, `\rvert`, `\mid`
> rather than as literal `|`, because a literal pipe inside a Markdown table cell is
> read as a column delimiter and breaks KaTeX rendering.

---

## How to read this book

The book has one destination: the **generalized Stokes' Theorem**

$$\int_{M}d\omega=\int_{\partial M}\omega,$$

and every chapter is machinery for stating it. Spivak's own framing (Preface) is that
the classical vector-calculus theorems of Green, Gauss and Stokes are a single theorem
in disguise, and that the disguise is the coordinate-bound notation of $\mathrm{grad}$,
$\mathrm{curl}$ and $\mathrm{div}$.

The route is short and steep — 146 pages — in three steps:

1. **Differentiation done right** (Ch. 1–2). The derivative $Df(a)$ is a *linear map*,
   defined by an approximation condition, not an array of partials. Partial derivatives
   come later and are a computational device.
2. **Integration done right** (Ch. 3). Riemann integration on rectangles, then the two
   tools that globalize it: **measure zero** (which characterizes integrability, via
   Lebesgue's criterion) and **partitions of unity** (which patch local definitions into
   global ones).
3. **Forms** (Ch. 4–5). Alternating tensors give $k$-forms; the exterior derivative $d$
   generalizes $\mathrm{grad}/\mathrm{curl}/\mathrm{div}$ simultaneously; chains and then
   manifolds give something to integrate over; and Stokes' Theorem falls out twice —
   once for chains (Ch. 4), once for manifolds (Ch. 5).

The last section, "The Classical Theorems," then *recovers* Green, Gauss and the
classical Stokes as corollaries. That ordering is the argument of the book.

---

## Master symbol table

### Euclidean space and topology (Ch. 1)

| Symbol | Meaning | Page |
| --- | --- | --- |
| $\mathbb{R}^{n}$ | Euclidean $n$-space; a point is $x=(x^{1},\dots,x^{n})$ | 1 |
| $\lvert x\rvert$ | the **norm**; "length = norm" | 1 |
| $\langle x,y\rangle$ | the **inner product**; usual one $\sum x^{i}y^{i}$ | 2, 77 |
| $e_{1},\dots,e_{n}$ | the *usual* basis for $\mathbb{R}^{n}$ | 3 |
| $\pi^{i}$ | the $i$-th **projection function** | 11 |
| $\mathrm{interior}\,A,\ \mathrm{exterior}\,A,\ \partial A$ | interior, exterior, **boundary of a set** | 7 |
| $o(f,x)$ | the **oscillation** of $f$ at $x$ | 13 |

### Differentiation (Ch. 2)

| Symbol | Meaning | Page |
| --- | --- | --- |
| $Df(a)$ | the **derivative**: the *linear map* with $\lvert f(a+h)-f(a)-Df(a)h\rvert/\lvert h\rvert\rightarrow0$ | 16 |
| $f^{\prime}(a)$ | the **Jacobian matrix** — the matrix *of* $Df(a)$ in the usual bases | 17 |
| $D_{i}f(a)$ | the $i$-th **partial derivative** | 25 |
| $D_{i,j}f$ | higher-order (mixed) partials | 26 |
| $D_{v}f(a)$ | the **directional derivative** | 33 |
| $C^{\infty}$ | smooth; Spivak writes "differentiable $=C^{\infty}$" for forms | 26, 88 |
| $\nabla f,\ \mathrm{grad}\,f$ | the gradient (introduced late, as a *derived* object) | 96 |

### Integration (Ch. 3)

| Symbol | Meaning | Page |
| --- | --- | --- |
| $L(f,P),\ U(f,P)$ | **lower** and **upper sums** over a partition $P$ | 47 |
| $\underline{\int},\ \overline{\int}$ | **lower** and **upper integrals** | 58 |
| $\int_{A}f$ | the integral over a set | 48, 55 |
| $v(A)$ | the **volume** (content) of a rectangle or set | 47, 56 |
| $\chi_{A}$ | the **characteristic function** of $A$ | 55 |
| $\Phi$ | a **partition of unity**, subordinate to a cover | 63 |

### Tensors and forms (Ch. 4)

| Symbol | Meaning | Page |
| --- | --- | --- |
| $\mathcal{T}^{k}(V)$ | the space of $k$-**tensors** on $V$ (multilinear $V^{k}\rightarrow\mathbb{R}$) | 75 |
| $S\otimes T$ | the **tensor product** | 75 |
| $\Lambda^{k}(V)$ | **alternating** $k$-tensors — *but see the Addenda note below* | 78 |
| $\mathrm{Alt}(T)$ | the alternation operator, $\frac{1}{k!}\sum_{\sigma\in S_{k}}\mathrm{sgn}\,\sigma\cdot T(v_{\sigma(1)},\dots)$ | 78 |
| $\mathrm{sgn}\,\sigma$ | the **sign of a permutation**: $+1$ if even, $-1$ if odd | 78 |
| $\omega\wedge\eta$ | the **wedge product** | 79 |
| $\det$ | the determinant, as the motivating alternating $n$-tensor | 78 |
| $\mathbb{R}^{n}_{p},\ v_{p}$ | the **tangent space** at $p$, and a tangent vector there | 86 |
| $dx^{i}$ | the dual basis covector; the atoms of every form | 88–89 |
| $\omega$ | a **differential form** (a $k$-form field) | 88 |
| $d\omega$ | the **differential** (exterior derivative) | 91 |
| $f^{*}\omega$ | the **pullback** of a form along $f$ | 89 |
| $c,\ \partial c$ | a **chain** and its **boundary** | 97, 98 |
| $I^{k}$ | the **standard $k$-cube**; a **singular $k$-cube** is a map $c:I^{k}\rightarrow\mathbb{R}^{n}$ | 97 |
| $\int_{c}\omega$ | the integral of a form over a chain | 101 |

### Manifolds (Ch. 5)

| Symbol | Meaning | Page |
| --- | --- | --- |
| $M$ | a **manifold**; $\dim M$ its dimension | 109 |
| $\mathbb{H}^{n}$ | the **half-space**, used to define a manifold-with-boundary | 113 |
| $\partial M$ | the **boundary of a manifold-with-boundary** | 113 |
| $M_{p}$ | the tangent space of $M$ at $p$ | 115 |
| $\mu$ | an **orientation**; $\mu_{p}$ at a point | 117, 119 |
| $n(p)$ | the **outward unit normal** | 119 |
| $dV$ | the **volume element** of an oriented manifold | 83, 126 |
| $dA,\ ds$ | the element of area / of length | 126 |
| $\int_{M}\omega$ | the integral of a form over a manifold | 123–124 |

---

## Chapter 1 — Functions on Euclidean Space (p. 1)

**Norm and inner product (p. 1).** $\lvert x\rvert$, $\langle x,y\rangle$, the
**triangle inequality**, the **polarization identity**, **orthogonality**, angle and
angle-preserving maps, norm-preserving maps, bilinearity, symmetry and positive
definiteness of the inner product.

**Subsets of Euclidean space (p. 5).** Open and closed **rectangles**; open and closed
sets; interior, exterior and **boundary of a set**; **open cover**; **compact**; the
**Heine–Borel Theorem**.

**Functions and continuity (p. 11).** Functions, domain, graph; **component
functions**; composition; **projection functions**; limits and continuity; the
**oscillation** $o(f,x)$, which is the tool Ch. 3 uses to characterize the set of
discontinuities.

---

## Chapter 2 — Differentiation (p. 15)

**Basic definitions (p. 19).** The central move of the book:
$f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$ is **differentiable at $a$** if there is a
*linear map* $\lambda$ with

$$\lim_{h\rightarrow0}\frac{\lvert f(a+h)-f(a)-\lambda(h)\rvert}{\lvert h\rvert}=0,$$

and then $Df(a)\equiv\lambda$ is unique. "Equal up to $n$-th order" formalizes the
approximation idea. The **Jacobian matrix** $f^{\prime}(a)$ is the matrix representing
$Df(a)$ — a representation, not the definition.

**Basic theorems (p. 25).** Linearity; the **chain rule**
$D(g\circ f)(a)=Dg(f(a))\circ Df(a)$ — a *composition of linear maps*, which is why it
looks like matrix multiplication in coordinates.

**Partial derivatives (p. 30).** $D_{i}f$; maxima and minima; higher-order (mixed)
partials and the theorem that they commute under continuity; **continuously
differentiable** ($C^{1}$).

**Derivatives (p. 34).** The key implication: *continuous partials $\Rightarrow$
differentiable*. The converse fails, and the **directional derivative** $D_{v}f$ can
exist in every direction without $f$ being differentiable — Spivak is careful about
this. **Homogeneous functions**.

**Inverse functions (p. 34–39).** The **Inverse Function Theorem** (p. 35): if $Df(a)$
is invertible then $f$ is invertible near $a$, and $Df^{-1}(f(a))=[Df(a)]^{-1}$.
(Addendum 1 notes $f^{-1}$ is in fact $C^{\infty}$ when $f$ is, via Cramer's rule on
the entries of the inverse matrix.)

**Implicit functions (p. 41–44).** The **Implicit Function Theorem**, derived from the
inverse function theorem.

**Notation (p. 44).** A short, openly opinionated critique of classical
partial-derivative notation: $\partial f/\partial x$ forces irrelevant letters into the
statement of the chain rule, and "note that $f$ means something different on the two
sides of the equation!" Spivak's point is that $df/dx$ tempts one into meaningless
separate definitions of $df$ and $dx$ — and that Chapter 4 will supply *rigorous*
definitions making those manipulations theorems. He then leaves it to the reader to
decide whether the modern definitions are an improvement.

---

## Chapter 3 — Integration (p. 46)

**Basic definitions (p. 46).** Partitions of a closed rectangle; subrectangles;
refinement; lower and upper sums $L(f,P)$, $U(f,P)$; lower and upper integrals;
integrability.

**Measure zero and content zero (p. 50–52).** Two distinct notions (countable vs.
finite covers), carefully separated.

**Integrable functions (p. 52).** **Lebesgue's criterion**: a bounded $f$ is Riemann
integrable iff its set of discontinuities has **measure zero** — stated via the
oscillation $o(f,x)$ from Ch. 1. **Jordan-measurable** sets; integration over a set
using the characteristic function $\chi_{A}$; **content**.

**Fubini's Theorem (p. 56).** Reduction of a multiple integral to **iterated
integrals**, stated with upper/lower integrals so it survives functions not integrable
in each variable separately. **Cavalieri's principle** appears as an application.

**Partitions of unity (p. 63).** The globalization tool: a family $\Phi$
**subordinate** to an open cover, used to define $\int_{A}f$ for $A$ open and to make
local constructions global. Used again in Ch. 5 to define $\int_{M}\omega$.

**Change of variable (p. 67–72).** The change-of-variable theorem with the
$\lvert\det g^{\prime}\rvert$ factor; **polar coordinates**; **Sard's Theorem** (p. 72).

---

## Chapter 4 — Integration on Chains (p. 75)

**Algebraic preliminaries (p. 75).** Multilinear functions; **$k$-tensors**
$\mathcal{T}^{k}(V)$; the **tensor product** $S\otimes T$; the inner product as a
symmetric $2$-tensor and $\det$ as an alternating $n$-tensor. A $k$-tensor is
**alternating** if interchanging two arguments flips the sign; these form
$\Lambda^{k}(V)$. The **alternation operator**

$$\mathrm{Alt}(T)(v_{1},\dots,v_{k})=\frac{1}{k!}\sum_{\sigma\in S_{k}}\mathrm{sgn}\,\sigma\cdot T(v_{\sigma(1)},\dots,v_{\sigma(k)})$$

projects $\mathcal{T}^{k}$ onto $\Lambda^{k}$ ($\mathrm{Alt}\circ\mathrm{Alt}=\mathrm{Alt}$).
The **wedge product** (p. 79) is built from it. Then **orientation** (p. 82), the
**volume element** (p. 83), the **cross product** (p. 84) as a $3$-dimensional
accident, and self-adjointness (p. 85).

**Fields and forms (p. 86).** **Tangent space** $\mathbb{R}^{n}_{p}$; **vector
fields**; **differential forms** as alternating-tensor fields; $dx^{i}$; continuity and
differentiability of forms ("differentiable $=C^{\infty}$"); the **differential** $d$
(p. 91) and its properties ($d\circ d=0$); **closed** and **exact** forms (p. 92);
**star-shaped** sets and the **Poincaré Lemma** (p. 93–94): on a star-shaped set,
closed $\Rightarrow$ exact. **Divergence** and **curl** appear here (p. 88) as *special
cases of $d$*, which is the chapter's payoff.

**Geometric preliminaries (p. 97).** **Singular $n$-cubes** and the **standard
$n$-cube** $I^{n}$; **chains** as formal sums of cubes; **faces** of a singular cube;
the **boundary** operator $\partial$, with $\partial\circ\partial=0$ mirroring
$d\circ d=0$.

**The Fundamental Theorem of Calculus (p. 100–104).** **Stokes' Theorem for chains**,
$\int_{c}d\omega=\int_{\partial c}\omega$ (p. 102). Then **line integrals**, **surface
integrals**, **independence of parameterization**, and the **winding number**. The
chapter closes with complex applications: **Cauchy–Riemann equations**, the **Cauchy
Integral Theorem** and **Formula**, and the **Fundamental Theorem of Algebra**
(p. 105–106).

---

## Chapter 5 — Integration on Manifolds (p. 109)

**Manifolds (p. 109).** **Manifolds** defined by local diffeomorphism to
$\mathbb{R}^{k}$; **diffeomorphism**; **coordinate systems** and the **coordinate
condition**; **manifold-with-boundary** (p. 113) via the half-space $\mathbb{H}^{n}$,
with $\partial M$ the part landing on the boundary hyperplane. Examples: the sphere,
the torus, the Möbius strip. **Lagrange's method** / **Lagrangian multipliers** appear
here (p. 122) as a manifold statement about constraints.

**Fields and forms on manifolds (p. 115).** Tangent space $M_{p}$; vector fields and
forms on $M$; **orientable** manifolds; **orientation**, **consistent choices**,
**induced orientation** on $\partial M$; orientation-preserving maps. The Möbius strip
is the standing counterexample to orientability.

**Stokes' Theorem on manifolds (p. 122–124).** The theorem the book exists for:

$$\int_{M}d\omega=\int_{\partial M}\omega,$$

with $\int_{M}\omega$ defined using a partition of unity (Ch. 3) and the induced
orientation on $\partial M$.

**The volume element (p. 126).** $dV$ on an oriented manifold; element of **length**
and of **area**; **surface area**; **absolute** tensors and forms, which let one
integrate on *non*-orientable manifolds; the **outward unit normal**; the **solid
angle** and **generalized cone** (p. 131); **linking number** (p. 132);
manifolds-with-corners (p. 131).

**The classical theorems (p. 134).** **Green's Theorem**, the **Divergence (Gauss)
Theorem**, and the classical **Stokes' Theorem**, each obtained as a specialization.
The physical vocabulary — **buoyant force**, **incompressible** and **irrotational**
fluids, rotation of a field — appears here to show what the identities were for.

---

## Notation traps worth flagging

- **Spivak retracts his own $\Lambda^{k}(V)$ notation.** Addendum 4 states that "the
  notation $\Lambda^{k}(V)$ appearing in this book is incorrect, since it conflicts
  with the standard definition of $\Lambda^{k}(V)$ (as a certain quotient of the tensor
  algebra of $V$)." The space he means is naturally isomorphic to $\Lambda^{k}(V^{*})$
  for finite-dimensional $V$, and he suggests $\Omega^{k}(V)$ instead — *substitution
  to be made on pp. 78–85, 88–89, 116 and 126–128*. Anyone cross-reading this book with
  a modern text must apply that correction mentally.
- **$Df(a)$ vs. $f^{\prime}(a)$.** $Df(a)$ is the linear map; $f^{\prime}(a)$ is its
  matrix in the usual bases. Most later texts write $Df$ for both.
- **$D_{i}f$ is a partial, $D_{v}f$ is directional, $Df$ is the total derivative** —
  three different things sharing one letter. All directional derivatives can exist at a
  point where $Df$ does not.
- **"Differentiable" means $C^{\infty}$ for forms** (p. 88) but means the approximation
  condition for functions (p. 15). The word is overloaded by design and Spivak flags it.
- **Measure zero $\neq$ content zero** (pp. 50–52) — countable vs. finite covers. The
  distinction carries the integrability criterion.
- **$\partial$ is overloaded**: boundary of a *set* (p. 7), of a *chain* (p. 98), and of
  a *manifold-with-boundary* (p. 113). These do not coincide; the boundary of the closed
  unit ball as a set is the sphere, but the manifold boundary is a different construction.
- **Errata worth knowing.** Addendum 2 simplifies Theorem 3-8 and notes that *"the proof
  of the converse part contains an error"*; Addendum 3 amplifies the argument in
  Theorem 3-14 (Sard's Theorem). Read the Addenda (pp. 145–146) before relying on those
  proofs.
