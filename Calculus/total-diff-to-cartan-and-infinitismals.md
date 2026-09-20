## Cartan's theory of the total differential

**The starting point.** In multivariable calculus you meet the total differential of a function,

$$df = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy + \cdots$$

and you learn to ask the converse question: given an expression $\omega = P\,dx + Q\,dy$, is it the total differential of some $f$? The classical answer is a condition on cross-partials, $\partial P/\partial y = \partial Q/\partial x$. Élie Cartan's contribution was to see that this scattered collection of facts — exactness conditions, line and surface integrals, grad/curl/div, the change-of-variables formula, the integrability conditions for systems of differential equations — is one algebraic structure seen from several angles.

**Move 1: make differentials into an algebra.** Cartan treats the symbols $dx, dy, dz$ not as "infinitesimals" but as elements of a formal algebra with one rule: they *anticommute*.

$$dx\wedge dy = -\,dy\wedge dx, \qquad dx\wedge dx = 0$$

This is Grassmann's exterior algebra, and the sign rule is not arbitrary — it encodes *oriented area*. A $k$-form is a sum of terms $f\,dx^{i_1}\wedge\cdots\wedge dx^{i_k}$, and the natural thing to do with a $k$-form is integrate it over a $k$-dimensional oriented surface. Antisymmetry is exactly what makes the answer flip sign when you reverse orientation, and vanish when the "directions" are degenerate.

**Move 2: one derivative to replace them all.** Cartan defines the *exterior derivative* $d$, which takes $k$-forms to $(k{+}1)$-forms, by extending the total differential:

$$d\big(f\,dx^{i_1}\wedge\cdots\wedge dx^{i_k}\big) = df\wedge dx^{i_1}\wedge\cdots\wedge dx^{i_k}$$

In $\mathbb{R}^3$, applying $d$ to a 0-form gives the gradient, to a 1-form gives the curl, to a 2-form gives the divergence. Three theorems of vector calculus become three instances of one operator. (Historically, $d\omega$ for a 1-form is Frobenius's "bilinear covariant" from the Pfaff problem; Cartan's insight was that it is itself a form, so the construction iterates.)

**Move 3: $d\circ d = 0$.** This one identity is the theory's engine. It is just the equality of mixed partials $\partial^2 f/\partial x\partial y = \partial^2 f/\partial y\partial x$ combined with antisymmetry $dx\wedge dy = -dy\wedge dx$: the symmetric object meets the antisymmetric one and dies. It generalizes $\text{curl}(\text{grad}) = 0$ and $\text{div}(\text{curl}) = 0$.

So *every* total differential is closed ($d\omega = 0$), and the question "is $\omega$ a total differential?" becomes "is every closed form exact?" The **Poincaré lemma** says yes, locally — but globally the failure is real and measurable. On the punctured plane, $d\theta = (x\,dy - y\,dx)/(x^2+y^2)$ is closed but not exact; there is no single-valued angle function. That gap is the topology of the domain, and quantifying it is de Rham cohomology. Cartan's calculus is the bridge between differentiation and topology.

**Move 4: forms don't care about coordinates.** If $\phi$ is a smooth map, forms pull back along it ($\phi^*$) and — crucially — $d(\phi^*\omega) = \phi^*(d\omega)$. Exterior differentiation commutes with change of variables. This is why the Jacobian determinant in the change-of-variables formula appears automatically rather than being inserted by hand, and it is what lets the whole machinery live on manifolds where no global coordinates exist.

**The payoff: one Stokes theorem.** Green, Gauss, Kelvin–Stokes, and the fundamental theorem of calculus collapse into

$$\int_M d\omega = \int_{\partial M}\omega$$

Read as a pairing, $d$ and $\partial$ are adjoint: the boundary operator on domains and the derivative on forms are the same idea on opposite sides of the integral sign. And $d^2 = 0$ mirrors $\partial^2 = 0$ (a boundary has no boundary).

**Where Cartan took it.** His own motivation was the *Pfaff problem* — when does a system of equations $\omega^1 = \cdots = \omega^r = 0$ admit integral submanifolds of a given dimension? — answered by the Frobenius condition $d\omega^i \equiv 0$ modulo the $\omega^j$, and by his own far deeper Cartan–Kähler theory. He built on this the **method of moving frames**, where the geometry of a space is packaged in structure equations of the shape $d\omega = -\omega\wedge\omega + \Omega$, with $\Omega$ the curvature. That is the direct ancestor of gauge theory: $F = dA + A\wedge A$ is Cartan's structure equation.

**The one-sentence version.** Cartan turned the total differential from an operation on functions into a calculus on an algebra of oriented integrands, where the single rule $d^2 = 0$ simultaneously expresses equality of mixed partials, the vanishing of curl-of-grad, the integrability conditions for differential systems, and the fact that boundaries have no boundaries.

---

**Original sources**

- É. Cartan, *Sur certaines expressions différentielles et le problème de Pfaff*, Ann. Sci. École Norm. Sup. (3) **16** (1899), 239–332 — the founding paper; free at [Numdam](https://www.numdam.org/item/?id=ASENS_1899_3_16__239_0).
- É. Cartan, *Leçons sur les invariants intégraux*, Hermann, 1922 — the readable exposition, with the mechanics/Poincaré connection.
- É. Cartan, *Les systèmes différentiels extérieurs et leurs applications géométriques*, Hermann, 1945.
- G. Frobenius, *Über das Pfaffsche Problem*, J. reine angew. Math. **82** (1877), 230–315 — the bilinear covariant Cartan reinterpreted.
- H. Grassmann, *Die lineale Ausdehnungslehre*, 1844 — the exterior product.

**Historical and modern accounts**

- V. Katz, [*The history of differential forms from Clairaut to Poincaré*](https://www.sciencedirect.com/science/article/pii/0315086081900276), Historia Mathematica **8** (1981), 161–188.
- T. Hawkins, [*Frobenius, Cartan, and the Problem of Pfaff*](https://link.springer.com/article/10.1007/s00407-004-0095-4), Arch. Hist. Exact Sci. **59** (2005).
- S.-S. Chern & C. Chevalley, *Élie Cartan and his mathematical work*, Bull. AMS **58** (1952), 217–250.
- M. Spivak, *Calculus on Manifolds* (1965) — the compact undergraduate treatment; H. Flanders, *Differential Forms with Applications to the Physical Sciences* (1963) — the physicist's route in.

Sources: [Numdam — Cartan
1899](https://www.numdam.org/item/?id=ASENS_1899_3_16__239_0), [Hawkins,
Arch. Hist. Exact
Sci.](https://link.springer.com/article/10.1007/s00407-004-0095-4), [Katz,
Historia
Mathematica](https://www.sciencedirect.com/science/article/pii/0315086081900276)




## Two ways to make "$dx$" honest

Both traditions are answering the same embarrassment: freshman calculus writes $df = f'(x)\,dx$ and then refuses to say what $dx$ *is*. Cartan and Robinson give opposite answers, and the opposition is clean.

**Cartan eliminates the infinitesimal.** $dx$ is not small. It is a linear functional on tangent vectors — $dx(v) = v^1$, the first component. It has no size at all; it's a machine that eats a direction and returns a number. Smallness never enters, because the limit was already taken when you defined the tangent space.

**Robinson keeps the infinitesimal.** $dx$ is literally a number, an element of a proper ordered field extension $^*\mathbb{R} \supset \mathbb{R}$ containing $\varepsilon$ with $0 < |\varepsilon| < 1/n$ for every standard $n$. Then $f'(x) = \operatorname{st}\!\big(\Delta f/\Delta x\big)$, the standard part of an honest quotient of honest numbers, and $\int_a^b f\,dx$ is the standard part of a hyperfinite Riemann sum with infinitesimal mesh. The limit is not taken; it is replaced by a rounding map $\operatorname{st}: $ (finite hyperreals) $\to \mathbb{R}$.

## The engines are of different kinds

Robinson's engine is **logical**: the transfer principle (Łoś's theorem — every first-order statement true of $\mathbb{R}$ is true of $^*\mathbb{R}$), plus saturation. Cartan's engine is **algebraic-topological**: $d^2 = 0$ and naturality under pullback, $\phi^*d = d\,\phi^*$.

This matters for what each *produces*. Transfer is a licensing device: it tells you your infinitesimal argument was legitimate, but it does not hand you a new object. Nonstandard analysis is, in its standard formulations, largely a proof technique rather than new mathematics about standard objects — Nelson's Internal Set Theory is a conservative extension of ZFC (Powell's theorem, in the appendix of Nelson 1977), so anything it proves about standard sets was already provable. (Henson and Keisler later showed this is subtler in higher types, where some nonstandard frameworks do exceed the corresponding standard theory.) Cartan's calculus, by contrast, manufactures invariants that were not there before: cohomology classes, characteristic classes, obstructions.

## Where they part company in the *multivariable* case

This is the sharpest contrast, and it's usually underappreciated.

**Orientation and antisymmetry are built in for Cartan and absent for Robinson.** The wedge $dx \wedge dy = -dy \wedge dx$ is what makes $\int_M d\omega = \int_{\partial M}\omega$ work in all dimensions with correct signs. Hyperreals give you infinitesimal *rectangles* but no sign rule; you still choose orientations by hand exactly as in classical multivariable calculus. NSA rigorizes the limiting process; it does not reorganize the combinatorics of orientation.

**Yet NSA gives the best intuition for why Stokes' theorem is true.** Chop the region into a hyperfinite grid of infinitesimal cells; each interior face is shared by two cells and appears with opposite orientation, so the sum telescopes and only $\partial M$ survives. That cancellation *is* antisymmetry, discovered from the other side. So the two views are complementary on exactly this theorem: NSA supplies the mechanism, Cartan supplies the bookkeeping that makes it automatic.

**$d^2 = 0$ versus $d^2 f \neq 0$.** Here they genuinely disagree in appearance, and resolving it is instructive. In the infinitesimal picture, the second differential of $f$ is the Hessian quadratic form $\sum \partial_i\partial_j f\, dx^i dx^j$ — very much nonzero. In Cartan's calculus $d(df) = 0$. There's no contradiction: Cartan's $d$ antisymmetrizes, and the Hessian is symmetric, so $d^2$ annihilates it. Equality of mixed partials is exactly the identity $d^2=0$. They are different operators on differentials, and NSA keeps the one Cartan discards.

**Global structure.** $d^2 = 0$ gives closed-versus-exact, hence de Rham cohomology, hence a computable measure of how a domain fails to be simply connected. Nonstandard analysis has no analogue. It is a foundation for local analysis, agnostic about topology.

## A comparison table

| | Cartan / exterior calculus | Robinson / hyperreals |
|---|---|---|
| What $dx$ is | a covector (linear functional) | an infinitesimal number in $^*\mathbb{R}$ |
| Size of $dx$ | undefined; not a magnitude | genuinely nonzero, smaller than every $1/n$ |
| Core principle | $d^2=0$, pullback naturality | transfer (Łoś) + saturation + standard part |
| Number system | $\mathbb{R}$ unchanged | $\mathbb{R}$ replaced by a nonarchimedean field |
| Logic | classical | classical |
| Cost | must learn multilinear algebra | needs a nonprincipal ultrafilter (AC), or IST's axioms |
| Orientation | built into $\wedge$ | handled by hand |
| Produces new invariants | yes (cohomology) | no (conservative, in the standard settings) |
| Killer application | gauge theory, topology, geometry | measure theory, stochastic analysis, asymptotics |
| Higher differentials | $d^2f=0$ | $d^2f=$ Hessian, second-order infinitesimal |

## Where NSA genuinely earns its keep

Not in multivariable calculus, honestly — but in measure and probability. **Loeb's construction** turns a hyperfinite internal measure into a genuine standard $\sigma$-additive measure, and **Anderson** used it to build Brownian motion literally as a hyperfinite random walk with infinitesimal steps, recovering the Itô integral as a hyperfinite sum. That is the case where the infinitesimal picture is not just intuition-preserving but computationally clarifying: continuous-time stochastic processes become finite combinatorics.

## The synthesis you might actually want

There *is* a framework where infinitesimals give you Cartan's calculus rather than competing with it: **synthetic differential geometry** (Lawvere, Kock). Its infinitesimals are *nilpotent* — $D = \{d : d^2 = 0\}$ — rather than invertible like Robinson's. A tangent vector is literally a map $D \to M$; a $k$-form is a function on infinitesimal $k$-cubes; the Kock–Lawvere axiom makes Taylor expansion exact rather than approximate; and the wedge relation $dx \wedge dx = 0$ stops looking like a formal convention and starts looking like what it is — nilpotence. The price is that you must work in a topos with intuitionistic logic: you cannot assert $d = 0$ or $d \neq 0$ for a nilpotent, so excluded middle goes. That is the deep divide: **Robinson buys classical logic at the cost of a strange number system; Lawvere buys a natural number system at the cost of classical logic.** Cartan sidesteps both by refusing to have infinitesimals at all.

## Which to reach for

For anything geometric — latent-space topology, normalizing flows (where the change-of-variables log-det is precisely pullback of a volume form), Hodge theory on graphs and simplicial complexes, information geometry, curvature of representation manifolds — Cartan is the working language, and there is no NSA substitute. For anything where you want to replace a limit with a literal finite-but-huge object — hyperfinite width limits, discrete-to-continuum arguments, stochastic processes — the nonstandard framing can be a real economy of thought, though it remains a minority dialect.

---

**Nonstandard analysis — originals**

- A. Robinson, *Non-standard analysis*, Proc. Kon. Ned. Akad. Wet. A **64** / Indag. Math. **23** (1961), 432–440 — the announcement.
- A. Robinson, *Non-standard Analysis*, North-Holland, 1966 — the book.
- E. Nelson, [*Internal set theory: a new approach to nonstandard analysis*](https://www.ams.org/journals/bull/1977-83-06/S0002-9904-1977-14398-X/), Bull. AMS **83** (1977), 1165–1198 — axiomatic route, no ultrafilters; contains Powell's conservativity proof.
- P. Loeb, [*Conversion from nonstandard to standard measure spaces and applications in probability theory*](https://www.ams.org/journals/tran/1975-211/S0002-9947-1975-0390154-8/), Trans. AMS **211** (1975), 113–122.
- R. M. Anderson, [*A non-standard representation for Brownian motion and Itô integration*](https://link.springer.com/article/10.1007/BF02756559), Israel J. Math. **25** (1976), 15–46.
- C. W. Henson & H. J. Keisler, [*On the strength of nonstandard analysis*](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/C927D0C878BBCB80B209E3500898820B/S0022481200031248a.pdf/on_the_strength_of_nonstandard_analysis.pdf), J. Symbolic Logic **51** (1986), 377–386.
- H. J. Keisler, *Elementary Calculus: An Infinitesimal Approach*, 1976 — free from the author; the actual undergraduate text, mostly single-variable.

**Synthetic / nilpotent infinitesimals**

- F. W. Lawvere, *Categorical dynamics* (1967 Chicago lectures) — the origin.
- A. Kock, *Synthetic Differential Geometry*, 2nd ed., CUP 2006 — free from the author's page.
- J. L. Bell, *A Primer of Infinitesimal Analysis*, CUP — the gentlest entry.
- I. Moerdijk & G. Reyes, *Models for Smooth Infinitesimal Analysis*, Springer 1991 — shows the models exist.

Sources: [Henson & Keisler, JSL 1986](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/C927D0C878BBCB80B209E3500898820B/S0022481200031248a.pdf/on_the_strength_of_nonstandard_analysis.pdf), [Anderson, Israel J. Math 1976](https://link.springer.com/article/10.1007/BF02756559), [Keisler publication list](https://people.math.wisc.edu/~hkeisler/papers.html)
