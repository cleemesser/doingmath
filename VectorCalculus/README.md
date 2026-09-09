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

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | [`Div_Curl_and_the_Jacobian`](Div_Curl_and_the_Jacobian.py) | Splits $DF$ into **dilation ⊕ strain ⊕ spin** — the decomposition of $\mathfrak{gl}(n)$ into irreducible $SO(n)$ representations — and identifies $\operatorname{div}F=\operatorname{tr}DF$ with the first summand and $\operatorname{curl}F=2\,\mathrm{vee}(\operatorname{skew}DF)$ with the third. Then: Liouville ($\operatorname{div}$ = log-rate of volume change), Cauchy–Stokes ($\operatorname{curl}$ = twice the mean angular velocity of material line elements), the Spivak/forms picture ($d\iota_F\mu=(\operatorname{div}F)\mu$ and $dF^\flat=2A$, proven symbolically), Stokes' theorem as the coordinate-free *definition* (with the $O(r^2)$ error coefficient predicted and checked), why curl is a vector only when $n=3$, and what the two operators are blind to. |

## The thread

**Div and curl are two of the three irreducible pieces of $DF$, not a complete description of it.**

$$DF \;=\; \underbrace{\tfrac{1}{n}(\operatorname{tr}DF)I}_{\text{divergence}}
        \;\oplus\; \underbrace{S_0}_{\text{rate of strain — invisible}}
        \;\oplus\; \underbrace{A}_{\text{curl}},
\qquad n^2 = 1 + \left(\tfrac{n(n+1)}{2}-1\right) + \tbinom{n}{2}.$$

In $\mathbb{R}^3$ that is $9 = 1 + 5 + 3$: the middle summand holds the *majority* of the components
and neither operator reports it. Two consequences the notebook makes concrete:

- **$F(x,y)=(y,x)$ has $\operatorname{div}=0$ and $\operatorname{curl}=0$ everywhere, yet $DF\neq0$.**
  Pointwise, div and curl are far from determining the derivative. Globally they nearly do — that is
  Helmholtz, and the leftover ambiguity is harmonic, hence (by Hodge) topological.
- **$\operatorname{div}$ needs less structure than $\operatorname{curl}$.** Trace is a $GL$-invariant,
  so divergence survives *any* change of frame and needs only a volume form. The transpose — hence the
  symmetric/antisymmetric split — is defined by the inner product, so curl needs a metric, and an
  orientation on top of that to be a vector rather than a 2-form. The notebook exhibits a pure
  rotation acquiring a strain part under a non-orthogonal change of coordinates while its divergence
  does not move.

The other half of the thread is Spivak's: there is one operator, $d$, applied in two degrees, and the
metric and orientation are what disguise it as two.

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

## Related notebooks in this repository

- [`GeometricLinearAlgebra/04_Volume_Determinant_Trace`](../GeometricLinearAlgebra/04_Volume_Determinant_Trace.py)
  — the linear case: $\operatorname{tr}T=\frac{d}{dt}\big\rvert_0\det(I+tT)$, $\det e^{tT}=e^{t\operatorname{tr}T}$.
  This directory is its nonlinear continuation.
- [`LieGroups/VectorField_View_Lie_Theory`](../LieGroups/VectorField_View_Lie_Theory.py) — flows, the
  Jacobi–Lie bracket, $\exp$ as a flow.
- [`LieGroups/SO3_Lie_Theory`](../LieGroups/SO3_Lie_Theory.py) — the hat map
  $\mathbb{R}^3\to\mathfrak{so}(3)$ that turns the antisymmetric part of $DF$ into $\tfrac12\operatorname{curl}F$.

## Workflow

```bash
uv run jupytext --sync VectorCalculus/Div_Curl_and_the_Jacobian.py             # code/markdown only
uv run jupytext --sync --execute VectorCalculus/Div_Curl_and_the_Jacobian.py   # + fresh figures
MPLBACKEND=Agg uv run python VectorCalculus/Div_Curl_and_the_Jacobian.py       # headless numerics check
```
