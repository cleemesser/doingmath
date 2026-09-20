# Vector Calculus: what $\operatorname{div}$ and $\operatorname{curl}$ say about $DF$

A vector field $F:\mathbb{R}^n\to\mathbb{R}^n$ is unusual among smooth maps. Its derivative at a
point is an endomorphism of the tangent space, $DF(p):T_p\mathbb{R}^n\to T_p\mathbb{R}^n$. An
endomorphism is a linear map from a space to itself. For a general $f:M\to N$ the derivative maps
between two
*different* spaces. Such a thing has no trace, no eigenvalues and no symmetric part. Everything
classical vector calculus does with div and curl follows from that one identification.

Each notebook *computes* what it describes and checks its own claims. Numbers are checked with
`np.allclose`. Where the claim is an identity for *all* fields, it is checked symbolically with
SymPy over generic functions. A symbolic check of that kind is a proof, not a spot check. The
sibling [`LieGroups/`](../LieGroups/README.md) and
[`GeometricLinearAlgebra/`](../GeometricLinearAlgebra/README.md) sets follow the same rule.

Each `*.py` file is the source of truth, in jupytext "percent" format. The paired `.ipynb` is
generated with `uv run jupytext --sync <file>.py`. Add `--execute` to embed fresh figures. See the
repo `CLAUDE.md`.

## Notebooks

Notebook 1 exists in three versions: 1, 1b and 1c. They share section numbering, prose and figures.
They differ in how the numbers are produced, and each one is worth running for a different reason.
Notebooks 2 and 3 are different topics. Notebook 2 takes one section of notebook 1 and expands it.
Notebook 3 steps back and asks what *all* the competing generalizations of curl are.

| # | Notebook | In one line |
|---|----------|-------------|
| 1 | [`Div_Curl_and_the_Jacobian`](Div_Curl_and_the_Jacobian.py) | Splits $DF$ into dilation, strain and spin, then finds div and curl inside it |
| 1b | [`Div_Curl_and_the_Jacobian_autodiff`](Div_Curl_and_the_Jacobian_autodiff.py) | The same notebook, with every derivative from automatic differentiation |
| 1c | [`Div_Curl_and_the_Jacobian_tensorly`](Div_Curl_and_the_Jacobian_tensorly.py) | The same content again, written against no particular array library |
| 2 | [`The_Antisymmetric_Part_of_DF`](The_Antisymmetric_Part_of_DF.py) | What replaces the curl vector when $n \neq 3$ |
| 3 | [`Generalizations_of_Curl`](Generalizations_of_Curl.py) | Seven properties of curl, and the different operator each one generalizes to |

### 1. Div_Curl_and_the_Jacobian

This notebook splits $DF$ into dilation, strain and spin. It identifies
$\operatorname{div}F=\operatorname{tr}DF$ with the first piece, and
$\operatorname{curl}F=2\,\mathrm{vee}(\operatorname{skew}DF)$ with the third.

Gradient enters from the other end. $D(\nabla f)$ is the Hessian, which is symmetric by Clairaut's
theorem. So a gradient field has no spin part at all. That gives $\operatorname{curl}\nabla f=0$ and
$\operatorname{div}\nabla f=\operatorname{tr}\operatorname{Hess}f=\Delta f$.

It then covers Liouville's formula, where divergence is the log-rate of volume change. It covers the
Cauchy-Stokes reading, where curl is twice the mean angular velocity of material line elements. The
Spivak forms picture proves $d\iota_F\mu=(\operatorname{div}F)\mu$ and $dF^\flat=2A$ symbolically.
Stokes' theorem appears as the coordinate-free *definition*, with the $O(r^2)$ error coefficient
predicted and then checked. The last sections cover why curl is a vector only when $n=3$, what the
two operators are blind to, and a closing note. That note uses Schur's lemma to prove that div and
curl are the only operators of their kind.

### 1b. Div_Curl_and_the_Jacobian_autodiff

The same notebook, with every numerical derivative computed by automatic differentiation through
[`autograd`](https://github.com/HIPS/autograd) rather than finite differences. The derivatives are
exact to machine precision, and there is no step size.

Three results are available only in this version:

1. Clairaut's theorem as a measurement. The AD Hessian is not symmetric by construction, so its
   antisymmetric part is a genuine observation. Version 1's centered stencil forced the symmetry.
2. The flow Jacobian, found by differentiating the RK4 solver itself. It is compared against
   integrating the variational equation. The gap shrinks as $4^{-4}$ per step refinement, which is
   exactly the integrator's order.
3. Divergence without ever forming a Jacobian, through Jacobian-vector products. A scaling
   measurement shows where the stochastic estimator overtakes the exact one.

It also adds Appendix A, on forward mode versus reverse mode.

### 1c. Div_Curl_and_the_Jacobian_tensorly

The same content again, written against no particular array library. Every field is spelled in
[TensorLy](https://tensorly.org)'s backend-agnostic `tl.*`. One line, `BACKEND = "jax"` or
`BACKEND = "pytorch"`, decides what actually runs.

Note what TensorLy does and does not abstract. It covers the *array* layer. It has no `grad`,
`jacobian`, `hessian` or `jvp` anywhere in its API, and the notebook asserts this rather than
claiming it. So the differentiation layer is a small adapter written here. It dispatches on
`tl.get_backend()`.

It adds §0.5 on the two-layer design. That section includes a trap: TensorLy's numpy backend paired
with `autograd` returns a Jacobian of silent zeros rather than raising an error. Appendix B re-runs
the headline quantities under every installed backend. JAX and PyTorch come out identical bit for
bit, including one number that ends a 400-step RK4 integration.

### 2. The_Antisymmetric_Part_of_DF

This notebook takes §7 of notebook 1, "curl is a vector only when $n=3$". It replaces the dimension
count with the object it was counting.

The spin bivector $A=\tfrac12(DF-DF^{\mathsf T})$ exists in every dimension and every signature. By
the real Schur form it is $\lfloor n/2\rfloor$ rotation planes at $\lfloor n/2\rfloor$ independent
speeds. The vector curl needs two coincidences at once. A bivector in $\mathbb{R}^{n\le3}$ must have
only one plane, which is measured by $\operatorname{Pf}A=0$. A plane in $\mathbb{R}^3$ must have a
normal line. $n=5$ has the second without the first.

Then it turns to spacetime. $F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu$ is that same
construction applied to the 4-potential, and $6=\binom42$ is why the field has six components. The
split into $\mathbf E$ and $\mathbf B$ is the $n=3$ repackaging done inside one observer's slice. A
rotor boost is applied and checked against Jackson §11.9, while $\mathbf E^2-\mathbf B^2$ and
$\mathbf E\cdot\mathbf B$ hold still. The second of those is the Pfaffian again.

Both wedge formalisms appear. One is an $n$-dimensional SymPy $d$, $\wedge$ and $\star$. The other
is geometric algebra, where $\nabla F=\nabla\cdot F+\nabla\wedge F$ makes div and curl two grades of
one product, in identical code for $n=2,3,4,5$ and for $\mathbb{R}^{1,3}$.

It closes with an aside on the octonionic 7-dimensional cross product. That product is real, but it
is a rank-7 projection of a 21-dimensional space, and it is not $SO(7)$-equivariant. Averaged over
Haar $SO(7)$ it vanishes into the Monte-Carlo floor, so Schur still forbids a curl there.

### 3. Generalizations_of_Curl

This notebook asks the question notebook 2 answers only for its own candidate. What are the
generalizations of curl in $\mathbb{R}^n$, in the plural? The premise is that in $\mathbb{R}^3$
seven inequivalent properties each pin down $\nabla\times$ on their own, and all seven are verified
in §1. So generalizing means choosing which property to keep, and the choices diverge.

Keep "$d$ on a 1-form" and there were never three operators. $\mathbb{R}^n$ has $n$ of them, in the
de Rham complex, with dimensions $\binom nk$. Grad, curl and div are what that chain looks like when
$n=3$ makes it short enough to name every arrow. $\mathbb{R}^4$ has *two* middle operators, and
neither one is the curl. Here $d^2=0$ is proven in all degrees for $n=4,5$, and generalized Stokes
is checked on a 2-cube in $\mathbb{R}^4$.

Keep "vector in, vector out" and you get nothing. §5 solves the equivariance condition
$L(RJR^{\mathsf T})=RL(J)$ directly. It finds a solution space of dimension 0 for every $n\neq3$,
and of dimension exactly 1 at $n=3$. That is stronger than the usual $\binom n2=n$ count, which
rules out only one construction. The consolation prize is that $\star dF^\flat$ becomes a vector
only after $n-3$ arbitrary directions are chosen.

Keep the curl-curl identity and you get the codifferential $\delta$ and $\Delta=d\delta+\delta d$.
That is the one operator here that returns what it consumes, in every dimension.

Keep the Poincaré lemma and you get topology. The angle form witnesses closed-but-not-exact, with
$\oint=2\pi$. Then Hodge's theorem is computed. Betti numbers of a torus, a sphere and an annulus
are obtained twice. The first count uses the ranks of $\partial$. The second uses $\dim\ker$ of the
combinatorial Hodge Laplacian. The two counts agree, and Euler-Poincaré falls out. The notebook ends
in a scorecard where no row scores seven.

## The thread

Div and curl read two of the three pieces of $DF$. They are not a complete description of it.

$$DF \;=\; \underbrace{\tfrac{1}{n}(\operatorname{tr}DF)I}_{\text{divergence}}
        \;+\; \underbrace{S_0}_{\text{rate of strain, invisible}}
        \;+\; \underbrace{A}_{\text{curl}},
\qquad n^2 = 1 + \left(\tfrac{n(n+1)}{2}-1\right) + \tbinom{n}{2}.$$

In $\mathbb{R}^3$ that is $9 = 1 + 5 + 3$. The middle piece holds most of the components, and
neither operator reports it. Three consequences the notebook makes concrete:

- $F(x,y)=(y,x)$ has $\operatorname{div}=0$ and $\operatorname{curl}=0$ everywhere, yet $DF\neq0$.
  Point by point, div and curl are far from determining the derivative. Globally they almost do,
  which is Helmholtz. The leftover ambiguity is harmonic, so by Hodge it is topological.
- $\operatorname{div}$ needs less structure than $\operatorname{grad}$ or $\operatorname{curl}$.
  Every invertible change of frame leaves the trace alone, so divergence needs only a notion of
  volume. The inner product defines the transpose, and so the symmetric/antisymmetric split. So curl
  needs a metric, plus an orientation before it can be a vector at all. Gradient meets the same fact
  one step earlier. $Df$ is a linear functional, and turning it into an arrow takes an inner
  product. So the *same* function has different gradients under different metrics, as
  $\nabla_{\!M}f=M^{-1}\nabla f$. The notebook shows both. A gradient arrow moves when the metric
  changes. A pure rotation picks up a strain part under a non-orthogonal change of coordinates,
  while its divergence stays put.
- A gradient field is exactly the case $A=0$. $D(\nabla f)=\operatorname{Hess}f$ is symmetric, so
  $\operatorname{curl}\nabla f=0$ is not a computation. It is the symmetry of second partials. And
  $\operatorname{div}\nabla f=\Delta f$ makes the Laplacian the trace of the Hessian, which is the
  same projection one step along. A harmonic $f$ removes the dilation piece too, so only strain is
  left. That is where the "invisible" fields above come from.

The other half of the thread is Spivak's. There is one operator, $d$, used in three degrees. The
metric and the orientation are what disguise it as grad, curl and div.

A closing section, §10 and only §10, names the structure in the language of group representations.
It writes $\mathfrak{gl}(n)=\mathbb{R}I\oplus\operatorname{Sym}_0(n)\oplus\mathfrak{so}(n)$ as
irreducible $SO(n)$ representations. It then uses Schur's lemma to turn "natural" into "unique".
Divergence is the only rotation-invariant scalar, and curl the only equivariant vector, that a first
derivative of a vector field can produce. Both multiplicities are *measured* numerically. The method
averages the conjugation action over random rotations and reads off a rank. Sections 1 to 9 need
only a first undergraduate linear algebra course, and never use that vocabulary.

## Which version to read

None of the three is the "real" one. They fail in different places, and that is the point.

- Finite differences ask nothing of the function except that you can evaluate it. So they work on a
  black box, a table of measurements, or a simulator you cannot see inside. They cost you a step
  size. Too large and truncation error dominates, too small and cancellation does. For a second
  derivative the best you can do is roughly $\sqrt[3]{\varepsilon}$ relative accuracy.
- Automatic differentiation gives the exact derivative *of the program*, with no step size. In
  return it demands that the program be written in its own numpy (`anp`) and stay free of in-place
  assignment. It also differentiates the code you actually wrote. That is a feature when you want
  the derivative of your discretization. It is a trap when you wanted the derivative of the equation
  the discretization approximates. §4 of the autodiff notebook puts both side by side.
- Symbolic differentiation (SymPy) is a third thing again, and all three notebooks use it unchanged.
  Only a symbolic derivative over generic functions can prove an identity for *all* $f$, which is
  what §2 and §6 need. AD evaluates at a point. It proves nothing.

One lesson belongs to the third version in particular. A backend abstraction abstracts an interface,
not a capability. `tl.sin` looks the same on every backend, but only some backends can be
differentiated. The one combination that cannot, numpy paired with `autograd`, fails by returning
zeros rather than raising an error. So the adapter in `_tensorly.py` refuses the numpy backend
outright. That is the only safe design. A differentiation layer that quietly produces plausible
wrong numbers is worse than one that does not exist.

The three versions agree numerically wherever they overlap. The same
$\Delta(\operatorname{div}F) = -2.008597054518664$ and $\det D\varphi_T = 4.657903971532783$ come
out of finite differences, `autograd`, JAX and PyTorch. That is the best cross-check any of them
has.

## Reference reading

### The forms picture

- M. Spivak, *Calculus on Manifolds*, Benjamin, 1965. Ch. 4 covers forms, the Poincaré lemma and
  Stokes' theorem for chains. Ch. 5 covers integration on manifolds, and its closing section derives
  Green's, Gauss's and Stokes's theorems as one theorem.
- H. Flanders, *Differential Forms with Applications to the Physical Sciences*, Academic Press, 1963.
- V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer, 1989. Ch. 7 covers
  forms. §16 covers $\mathcal{L}_F\mu=(\operatorname{div}F)\mu$.
- T. Frankel, *The Geometry of Physics*, 3rd ed., CUP, 2011. This covers what needs a metric and
  what does not.
- T. Needham, *Visual Differential Geometry and Forms*, Princeton, 2021.

### The originals

- J. C. Maxwell, "On the mathematical classification of physical quantities", *Proc. London Math.
  Soc.* **s1-3** (1871), 224–233. This is where *curl*, *convergence* and *slope* were named.
- G. G. Stokes, "On the theories of the internal friction of fluids in motion…", *Trans. Camb. Phil.
  Soc.* **8** (1845), 287–319. §1 has the dilation, strain and rotation decomposition of relative
  motion.
- H. Helmholtz, "Über Integrale der hydrodynamischen Gleichungen, welche den Wirbelbewegungen
  entsprechen", *J. reine angew. Math.* **55** (1858), 25–55. Tait's translation is in *Phil. Mag.*
  **33** (1867), 485–512.
- J. Liouville, "Sur la théorie de la variation des constantes arbitraires", *J. Math. Pures Appl.*
  **3** (1838), 342–349.
- G. K. Batchelor, *An Introduction to Fluid Dynamics*, CUP, 1967. See §2.3 for the paddle-wheel
  reading of Stokes 1845.

### For the closing note (§10)

- H. Weyl, *The Classical Groups*, Princeton, 1939. This covers tensor representations of the
  orthogonal group.
- W. Fulton and J. Harris, *Representation Theory: A First Course*, Springer, 1991. See §§1–3 for
  Schur's lemma and multiplicity counting.

## Related notebooks in this repository

- [`GeometricLinearAlgebra/04_Volume_Determinant_Trace`](../GeometricLinearAlgebra/04_Volume_Determinant_Trace.py)
  covers the linear case, $\operatorname{tr}T=\frac{d}{dt}\big\rvert_0\det(I+tT)$ and
  $\det e^{tT}=e^{t\operatorname{tr}T}$. This directory is its nonlinear continuation.
- [`LieGroups/VectorField_View_Lie_Theory`](../LieGroups/VectorField_View_Lie_Theory.py) covers
  flows, the Jacobi-Lie bracket, and $\exp$ as a flow.
- [`LieGroups/SO3_Lie_Theory`](../LieGroups/SO3_Lie_Theory.py) covers the hat map from
  $\mathbb{R}^3$ to antisymmetric matrices. That map turns the antisymmetric part of $DF$ into
  $\tfrac12\operatorname{curl}F$.

## Workflow

```bash
uv sync                                                                          # brings in autograd
uv run jupytext --sync VectorCalculus/Div_Curl_and_the_Jacobian.py               # code/markdown only
uv run jupytext --sync --execute VectorCalculus/Div_Curl_and_the_Jacobian.py     # + fresh figures
MPLBACKEND=Agg uv run python VectorCalculus/Div_Curl_and_the_Jacobian.py         # headless check
MPLBACKEND=Agg uv run python VectorCalculus/Div_Curl_and_the_Jacobian_autodiff.py
MPLBACKEND=Agg uv run python VectorCalculus/Div_Curl_and_the_Jacobian_tensorly.py
MPLBACKEND=Agg uv run python VectorCalculus/The_Antisymmetric_Part_of_DF.py
MPLBACKEND=Agg uv run python VectorCalculus/Generalizations_of_Curl.py
```

The TensorLy version needs `tensorly` plus at least one differentiable backend. JAX is the default
and is a base dependency. PyTorch lives behind the `torch` extra, installed with
`uv sync --extra torch`. Appendix B skips any backend it cannot import.
