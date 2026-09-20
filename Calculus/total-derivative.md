# Rules for the Total Derivative

## Core object

For $f:\mathbb{R}^n\to\mathbb{R}^m$ differentiable at $\mathbf{x}$, the total derivative $Df(\mathbf{x})$ is the *linear map* satisfying

$$f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x}) + Df(\mathbf{x})\mathbf{h} + o(\|\mathbf{h}\|).$$

Its matrix in standard coordinates is the Jacobian $J_{ij} = \partial f_i/\partial x_j$. Every rule below follows from this definition.

## Rules

**1. Chain rule.** For a curve $\mathbf{x}(t)$,
$$\frac{df}{dt} = \sum_i \frac{\partial f}{\partial x_i}\frac{dx_i}{dt} = \nabla f\cdot\dot{\mathbf{x}}.$$
General composition: $D(g\circ f)(\mathbf{x}) = Dg(f(\mathbf{x}))\,Df(\mathbf{x})$ — a matrix product, so order matters.

**2. Explicit time dependence.** If $f = f(\mathbf{x}(t),t)$,
$$\frac{df}{dt} = \frac{\partial f}{\partial t} + \sum_i \frac{\partial f}{\partial x_i}\dot{x}_i.$$
This is the material derivative $D/Dt = \partial_t + \mathbf{v}\cdot\nabla$ when $\dot{\mathbf{x}}$ is a velocity field.

**3. Differential form.** $df = \sum_i \frac{\partial f}{\partial x_i}\,dx_i$, coordinate-free once $dx_i$ is read as a basis covector. Invariant under change of coordinates.

**4. Linearity.** $D(af+bg) = a\,Df + b\,Dg$.

**5. Product and quotient.** $d(fg) = f\,dg + g\,df$ and $d(f/g) = (g\,df - f\,dg)/g^2$. For a bilinear map $B$,
$$D\,B(u,v)\,h = B(Du\,h,\;v) + B(u,\;Dv\,h),$$
which covers dot products, matrix products, and wedge products.

**6. Implicit differentiation.** If $F(x,y)=0$ with $\partial F/\partial y \neq 0$, then $dy/dx = -F_x/F_y$. Vector version: if $D_{\mathbf{y}}F$ is invertible, $D\mathbf{y}(\mathbf{x}) = -(D_{\mathbf{y}}F)^{-1}D_{\mathbf{x}}F$ (implicit function theorem).

**7. Inverse.** $Df^{-1}(f(\mathbf{x})) = [Df(\mathbf{x})]^{-1}$ wherever the inverse exists.

## Caveats

- Existence of all partials does **not** imply total differentiability. Counterexample: $f(x,y) = xy/(x^2+y^2)$ with $f(0,0)=0$; the partials exist at the origin but $f$ is not even continuous there.
- Sufficient condition: partials exist and are *continuous* on a neighborhood $\Rightarrow$ $f\in C^1$, hence totally differentiable.
- Directional derivatives equal $Df(\mathbf{x})\mathbf{v}$ only when the total derivative exists; otherwise they may all exist yet fail to be linear in $\mathbf{v}$.
- Symmetry of second derivatives, $\partial_i\partial_j f = \partial_j\partial_i f$, requires continuity of the mixed partials (Clairaut–Schwarz).

## References

- M. Spivak, *Calculus on Manifolds*, Benjamin, 1965 — Ch. 2, the cleanest treatment of $Df$ as a linear map.
- W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw–Hill, 1976 — Thm. 9.21 ($C^1$ criterion), Thm. 9.28 (implicit function theorem).
- V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer, 1989 — Ch. 7, the differential-form viewpoint.
