# Vector Calculus — what $\operatorname{div}$ and $\operatorname{curl}$ say about $DF$

A vector field $F:\mathbb{R}^n\to\mathbb{R}^n$ is unusual among smooth maps: its derivative at a
point is an **endomorphism of the tangent space**, $DF(p):T_p\mathbb{R}^n\to T_p\mathbb{R}^n$. For a
general $f:M\to N$ the derivative maps between two *different* spaces and has no trace, no
eigenvalues, no symmetric part. Everything classical vector calculus does with div and curl follows
from that one identification.

Like the sibling [`LieGroups/`](../LieGroups/README.md) and
[`GeometricLinearAlgebra/`](../GeometricLinearAlgebra/README.md) sets, each notebook *computes* what
it describes and cross-checks its own claims — numerically with `np.allclose`, and where the claim is
an identity for *all* fields, symbolically with SymPy over generic functions, which is a proof rather
than a spot check.

Each `*.py` is the source of truth (jupytext "percent" format); the paired `.ipynb` is generated with
`uv run jupytext --sync <file>.py` (add `--execute` to embed fresh figures). See the repo `CLAUDE.md`.

## Notebooks

The same material exists in two versions. They share section numbering, prose and figures; they
differ in how the numbers are produced, and each is worth running for a different reason.

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | [`Div_Curl_and_the_Jacobian`](Div_Curl_and_the_Jacobian.py) | Splits $DF$ into **dilation ⊕ strain ⊕ spin** and identifies $\operatorname{div}F=\operatorname{tr}DF$ with the first piece and $\operatorname{curl}F=2\,\mathrm{vee}(\operatorname{skew}DF)$ with the third. **Gradient** enters from the other end: $D(\nabla f)$ is the Hessian, symmetric by Clairaut, so a gradient field has no spin part at all ($\operatorname{curl}\nabla f=0$) and $\operatorname{div}\nabla f=\operatorname{tr}\operatorname{Hess}f=\Delta f$. Then: Liouville ($\operatorname{div}$ = log-rate of volume change), Cauchy–Stokes ($\operatorname{curl}$ = twice the mean angular velocity of material line elements), the Spivak/forms picture ($d\iota_F\mu=(\operatorname{div}F)\mu$ and $dF^\flat=2A$, proven symbolically), Stokes' theorem as the coordinate-free *definition* (with the $O(r^2)$ error coefficient predicted and checked), why curl is a vector only when $n=3$, what the two operators are blind to, and a closing note proving via Schur's lemma that div and curl are the **only** operators of their kind. |
| 1b | [`Div_Curl_and_the_Jacobian_autodiff`](Div_Curl_and_the_Jacobian_autodiff.py) | The same notebook with every numerical derivative computed by **automatic differentiation** ([`autograd`](https://github.com/HIPS/autograd)) rather than finite differences — exact to machine precision, no step size. Three results become available only in this version: **Clairaut's theorem as a measurement** (the AD Hessian is not symmetric by construction, so its antisymmetric part is a genuine observation — version 1's centred stencil forced the symmetry); **the flow Jacobian by differentiating the RK4 solver itself**, compared against integrating the variational equation, with the gap shrinking as $4^{-4}$ per step refinement, i.e. exactly at the integrator's order; and **divergence without ever forming a Jacobian**, via Jacobian–vector products, with a scaling measurement showing where the stochastic estimator overtakes the exact one. Adds Appendix A on forward versus reverse mode. |

## The thread

**Div and curl read two of the three pieces of $DF$; they are not a complete description of it.**

$$DF \;=\; \underbrace{\tfrac{1}{n}(\operatorname{tr}DF)I}_{\text{divergence}}
        \;+\; \underbrace{S_0}_{\text{rate of strain — invisible}}
        \;+\; \underbrace{A}_{\text{curl}},
\qquad n^2 = 1 + \left(\tfrac{n(n+1)}{2}-1\right) + \tbinom{n}{2}.$$

In $\mathbb{R}^3$ that is $9 = 1 + 5 + 3$: the middle piece holds the *majority* of the components
and neither operator reports it. Three consequences the notebook makes concrete:

- **$F(x,y)=(y,x)$ has $\operatorname{div}=0$ and $\operatorname{curl}=0$ everywhere, yet $DF\neq0$.**
  Pointwise, div and curl are far from determining the derivative. Globally they nearly do — that is
  Helmholtz, and the leftover ambiguity is harmonic, hence (by Hodge) topological.
- **$\operatorname{div}$ needs less structure than $\operatorname{grad}$ or $\operatorname{curl}$.**
  The trace is unchanged by *every* invertible change of frame, so divergence needs only a notion of
  volume. The transpose — hence the symmetric/antisymmetric split — is defined by the inner product,
  so curl needs a metric, plus an orientation to be a vector at all. Gradient meets the same fact one
  step earlier: $Df$ is a linear functional, and turning it into an arrow requires an inner product,
  so the *same* function has different gradients under different metrics ($\nabla_{\!M}f=M^{-1}\nabla f$).
  The notebook exhibits both — a gradient arrow moving when the metric changes, and a pure rotation
  acquiring a strain part under a non-orthogonal change of coordinates while its divergence does not
  move.
- **A gradient field is exactly the case $A=0$.** $D(\nabla f)=\operatorname{Hess}f$ is symmetric, so
  $\operatorname{curl}\nabla f=0$ is not a computation but the symmetry of second partials; and
  $\operatorname{div}\nabla f=\Delta f$ makes the Laplacian the trace of the Hessian, the same
  projection one step along. Harmonic $f$ kills the dilation piece too, leaving pure strain — which is
  where the "invisible" fields above come from.

The other half of the thread is Spivak's: there is one operator, $d$, applied in three degrees, and
the metric and orientation are what disguise it as grad, curl and div.

A closing section (§10, and only §10) names the structure in group-representation language —
$\mathfrak{gl}(n)=\mathbb{R}I\oplus\operatorname{Sym}_0(n)\oplus\mathfrak{so}(n)$ as irreducible
$SO(n)$ representations — and uses Schur's lemma to upgrade "natural" to "unique": divergence is the
only rotation-invariant scalar and curl the only equivariant vector that a first derivative of a
vector field can produce. Both multiplicities are *measured* numerically, by averaging the conjugation
action over random rotations and reading off a rank. **Sections 1–9 need only a first undergraduate
linear algebra course** and never use that vocabulary.

## Reference reading

**The forms picture**

- M. Spivak, *Calculus on Manifolds*, Benjamin, 1965 — Ch. 4 (forms, the Poincaré lemma, Stokes'
  theorem for chains) and Ch. 5 (integration on manifolds; the closing section derives Green's,
  Gauss's and Stokes's theorems as one theorem).
- H. Flanders, *Differential Forms with Applications to the Physical Sciences*, Academic Press, 1963.
- V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer, 1989 — Ch. 7 on
  forms; §16 for $\mathcal{L}_F\mu=(\operatorname{div}F)\mu$.
- T. Frankel, *The Geometry of Physics*, 3rd ed., CUP, 2011 — what needs a metric and what does not.
- T. Needham, *Visual Differential Geometry and Forms*, Princeton, 2021.

**The originals**

- J. C. Maxwell, "On the mathematical classification of physical quantities", *Proc. London Math.
  Soc.* **s1-3** (1871), 224–233 — where *curl*, *convergence* and *slope* were named.
- G. G. Stokes, "On the theories of the internal friction of fluids in motion…", *Trans. Camb. Phil.
  Soc.* **8** (1845), 287–319 — §1 has the dilation/strain/rotation decomposition of relative motion.
- H. Helmholtz, "Über Integrale der hydrodynamischen Gleichungen, welche den Wirbelbewegungen
  entsprechen", *J. reine angew. Math.* **55** (1858), 25–55 (Tait's translation, *Phil. Mag.* **33**,
  1867, 485–512).
- J. Liouville, "Sur la théorie de la variation des constantes arbitraires", *J. Math. Pures Appl.*
  **3** (1838), 342–349.
- G. K. Batchelor, *An Introduction to Fluid Dynamics*, CUP, 1967 — §2.3, the paddle-wheel reading of
  Stokes 1845.

**For the closing note (§10)**

- H. Weyl, *The Classical Groups*, Princeton, 1939 — tensor representations of the orthogonal group.
- W. Fulton and J. Harris, *Representation Theory: A First Course*, Springer, 1991 — §§1–3 for Schur's
  lemma and multiplicity counting.

## Related notebooks in this repository

- [`GeometricLinearAlgebra/04_Volume_Determinant_Trace`](../GeometricLinearAlgebra/04_Volume_Determinant_Trace.py)
  — the linear case: $\operatorname{tr}T=\frac{d}{dt}\big\rvert_0\det(I+tT)$, $\det e^{tT}=e^{t\operatorname{tr}T}$.
  This directory is its nonlinear continuation.
- [`LieGroups/VectorField_View_Lie_Theory`](../LieGroups/VectorField_View_Lie_Theory.py) — flows, the
  Jacobi–Lie bracket, $\exp$ as a flow.
- [`LieGroups/SO3_Lie_Theory`](../LieGroups/SO3_Lie_Theory.py) — the hat map from $\mathbb{R}^3$ to
  antisymmetric matrices, which turns the antisymmetric part of $DF$ into $\tfrac12\operatorname{curl}F$.

## Finite differences or autodiff?

Neither version is the "real" one; they fail in different places, and seeing both is the point.

- **Finite differences** need nothing of the function but the ability to evaluate it, so they work on
  a black box, a table of measurements, or a simulator you cannot see inside. They cost a step size:
  too large and truncation error dominates, too small and cancellation does, and the best you can do
  for a second derivative is roughly $\sqrt[3]{\varepsilon}$ relative accuracy.
- **Automatic differentiation** gives the exact derivative *of the program*, with no step size, but
  demands that the program be written in its own numpy (`anp`) and stay free of in-place assignment.
  It also differentiates the code you actually wrote — which is a feature when you want the
  derivative of your discretization, and a trap when you wanted the derivative of the equation your
  discretization approximates. §4 of the autodiff notebook puts both side by side.
- **Symbolic differentiation** (SymPy) is a third thing again, and both notebooks use it unchanged:
  only a symbolic derivative over generic functions can prove an identity for *all* $f$, which is
  what §2 and §6 need. AD evaluates at a point; it proves nothing.

## Workflow

```bash
uv sync                                                                          # brings in autograd
uv run jupytext --sync VectorCalculus/Div_Curl_and_the_Jacobian.py               # code/markdown only
uv run jupytext --sync --execute VectorCalculus/Div_Curl_and_the_Jacobian.py     # + fresh figures
MPLBACKEND=Agg uv run python VectorCalculus/Div_Curl_and_the_Jacobian.py         # headless check
MPLBACKEND=Agg uv run python VectorCalculus/Div_Curl_and_the_Jacobian_autodiff.py
```
