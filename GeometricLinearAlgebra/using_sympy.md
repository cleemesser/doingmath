SymPy for linear algebra
 — see exact fractions and symbols rather than floating-point noise, and they can watch each step of an algorithm unfold. Here are concrete ways to use it, organized by what you're teaching.

**Matrices and basic operations.** Start by having students build matrices and vectors directly so the notation maps onto what they see on the board. SymPy keeps entries exact, which is pedagogically valuable because `Rational(1,3)` displays as a fraction rather than `0.333...`.

```python
from sympy import Matrix, Rational, symbols, init_printing
init_printing()  # renders matrices in nice math notation in Jupyter

A = Matrix([[2, -1], [1, 3]])
b = Matrix([1, 4])
A + A, 3*A, A*b, A.T   # addition, scalar mult, matrix-vector product, transpose
```

**Gaussian elimination and RREF.** One of the most useful teaching features is `rref()`, which returns the reduced row echelon form along with the pivot columns. This lets students check their hand computations. To teach the *process* rather than just the answer, you can use elementary row operations explicitly so each step is visible:

```python
M = Matrix([[1, 2, 1], [2, 4, 0], [1, 1, 3]])
M.rref()                              # final answer + pivots
M.elementary_row_op('n->n+km', row1=1, row2=0, k=-2)  # one step at a time
```

Walking through `elementary_row_op` (or just slicing and reassigning rows) is excellent for showing that row reduction is a sequence of reversible operations.

**Solving systems.** `linsolve` is ideal for teaching the connection between systems, matrices, and solution sets — and crucially it handles underdetermined systems by returning parametrized solutions, which is exactly where students struggle. You can present a system three ways and show they agree: as equations, as an augmented matrix, and as `Ax = b`.

```python
from sympy import linsolve
x, y, z = symbols('x y z')
linsolve([x + y + z - 6, 2*x - y + z - 3, x + 2*y - z - 2], x, y, z)
linsolve((A_aug,), x, y, z)   # augmented-matrix form shows the free-variable case
```

**Vectors, dot and cross products, and geometry.** For introducing vectors you can keep things simple with `Matrix` columns and the built-in `.dot()`, `.cross()`, and `.norm()` methods. Because everything is symbolic, you can do things like derive the angle between two vectors exactly, or show orthogonality by getting a clean zero.

```python
u = Matrix([1, 2, 2]); v = Matrix([2, 0, -1])
u.dot(v), u.cross(v), u.norm()
from sympy import acos, simplify
theta = acos(u.dot(v) / (u.norm()*v.norm()))   # exact angle
```

For a more physics-flavored course, SymPy's `sympy.vector` and `sympy.physics.vector` modules let you work with coordinate systems and basis vectors symbolically, which is nice for connecting to multivariable calculus later.

**Eigenvalues, eigenvectors, and diagonalization.** This is where symbolic computation really shines pedagogically, because students can see the characteristic polynomial factor exactly and get eigenvectors as clean integer or rational vectors rather than normalized decimals.

```python
A = Matrix([[2, 0, 0], [1, 2, 1], [-1, 0, 1]])
A.charpoly()          # characteristic polynomial, great for showing det(A - λI)
A.eigenvals()         # eigenvalues with algebraic multiplicities
A.eigenvects()        # eigenvalues, multiplicities, and basis eigenvectors
P, D = A.diagonalize()  # show A = P D P^-1 explicitly
```

Having students verify `P*D*P.inv() == A` themselves reinforces what diagonalization actually means.

**Symbolic exploration with parameters.** A distinctive teaching move is to put a symbol *inside* a matrix and ask when something changes. For example, compute a determinant with a parameter and solve for when the matrix becomes singular — this connects determinants, invertibility, and eigenvalues in one exercise.

```python
k = symbols('k')
B = Matrix([[1, k], [k, 4]])
from sympy import solve
solve(B.det(), k)     # values of k where B is not invertible
```

**Bases, span, and subspaces.** SymPy has direct methods for the "four fundamental subspaces" discussion: `A.columnspace()`, `A.nullspace()`, `A.rowspace()`, and `A.rank()`. Letting students compute a null space and then verify `A * (null vector) == 0` makes the abstract definition concrete.

A few practical tips for the classroom. Use `init_printing()` (or run in Jupyter) so matrices render in proper math notation — this matters a lot for readability. Lean on exact arithmetic (`Rational`, `sqrt`, symbols) as the default and only call `.evalf()` or convert to floats when you specifically want to discuss numerical issues. And for algorithms like elimination, prefer the step-by-step methods over the one-shot answer so students see the procedure rather than treating SymPy as a black box. A natural capstone is to have students implement a piece of an algorithm themselves (say, one Gaussian-elimination step) and then check it against SymPy's built-in result.

If you tell me the level of your course (intro vs. proof-based, or whether it's paired with a programming component), I can tailor a set of specific worked exercises or a Jupyter notebook outline.


## Tensors
- notes from claude research

Yes, SymPy has fairly substantial tensor support, spread across a few different modules that serve different purposes. The tensor functionality is organized into N-dimensional arrays, indexed objects, and a full abstract tensor module, plus additional differential-geometry and physics tooling.

**N-dimensional arrays (`sympy.tensor.array`).** This is the most accessible entry point and the closest analog to a NumPy array but with symbolic entries. You get dense and sparse, mutable and immutable variants, and the usual operations: tensor product, contraction, element-wise application, derivatives, and reshaping.

```python
from sympy import symbols
from sympy.tensor.array import Array, tensorproduct, tensorcontraction, derive_by_array

x, y, z = symbols('x y z')
A = Array([[x, y], [z, x*y]])      # a 2x2 symbolic array
B = Array([1, 2])
tensorproduct(A, B)                # outer product -> higher-rank array
tensorcontraction(A, (0, 1))       # contract axes 0 and 1 (the trace here)
derive_by_array([x*y, x+z], [x, y, z])  # Jacobian-style differentiation
```

This is great for symbolic multilinear algebra and for differentiating expressions with respect to vectors/matrices.

**Indexed objects (`sympy.tensor.indexed`).** This module provides `IndexedBase`, `Indexed`, and `Idx`, which let you write things in abstract index notation like `A[i, j]*x[j]` without committing to concrete values. It's primarily symbolic bookkeeping — useful for deriving expressions and for code generation — but it does not by itself enforce Einstein summation or tensor transformation rules.

**Abstract tensors (`sympy.tensor.tensor`).** This is the most "real" tensor module in the differential-geometry/physics sense. It supports tensors defined over `TensorIndexType`s with a metric, covariant and contravariant indices, automatic index raising/lowering, Einstein summation via repeated indices, symmetry properties of indices, and canonicalization of tensor expressions. There's also a `toperators` submodule for things like partial-derivative operators on tensors. This is what you'd reach for if you want genuine index gymnastics rather than just arrays of expressions.

**Physics and differential geometry.** Beyond the core tensor module, `sympy.diffgeom` handles manifolds, coordinate systems, and differential forms, and the physics submodules include vector/dyadic machinery (`sympy.physics.vector`) and mechanics tooling. For general relativity specifically, people often compute Christoffel symbols, the Riemann and Ricci tensors, etc., by combining the N-dim arrays with `derive_by_array` and contractions — and there are well-known third-party libraries (like EinsteinPy and GraviPy) built on top of SymPy that package this more conveniently.

A couple of honest caveats. The different modules don't fully interoperate — the array module, the indexed module, and the abstract tensor module each have their own conventions, so you generally pick the one that fits your task rather than mixing them. The abstract `tensor` module has a steeper learning curve and somewhat terse documentation. And SymPy tensors are symbolic, so they're excellent for derivation and exact results but slow for large numerical work, where NumPy or specialized libraries are more appropriate.

For teaching purposes, the N-dimensional `Array` plus `tensorproduct`/`tensorcontraction`/`derive_by_array` is usually the sweet spot: it's intuitive, stays symbolic and exact, and lets students see tensor products and contractions concretely before introducing the heavier abstract-index machinery. If you let me know whether you're aiming at physics/Yes, SymPy has fairly substantial tensor support, spread across a few different modules that serve different purposes. The tensor functionality is organized into N-dimensional arrays, indexed objects, and a full abstract tensor module, plus additional differential-geometry and physics tooling.

**N-dimensional arrays (`sympy.tensor.array`).** This is the most accessible entry point and the closest analog to a NumPy array but with symbolic entries. You get dense and sparse, mutable and immutable variants, and the usual operations: tensor product, contraction, element-wise application, derivatives, and reshaping.

```python
from sympy import symbols
from sympy.tensor.array import Array, tensorproduct, tensorcontraction, derive_by_array

x, y, z = symbols('x y z')
A = Array([[x, y], [z, x*y]])      # a 2x2 symbolic array
B = Array([1, 2])
tensorproduct(A, B)                # outer product -> higher-rank array
tensorcontraction(A, (0, 1))       # contract axes 0 and 1 (the trace here)
derive_by_array([x*y, x+z], [x, y, z])  # Jacobian-style differentiation
```

This is great for symbolic multilinear algebra and for differentiating expressions with respect to vectors/matrices.

**Indexed objects (`sympy.tensor.indexed`).** This module provides `IndexedBase`, `Indexed`, and `Idx`, which let you write things in abstract index notation like `A[i, j]*x[j]` without committing to concrete values. It's primarily symbolic bookkeeping — useful for deriving expressions and for code generation — but it does not by itself enforce Einstein summation or tensor transformation rules.

**Abstract tensors (`sympy.tensor.tensor`).** This is the most "real" tensor module in the differential-geometry/physics sense. It supports tensors defined over `TensorIndexType`s with a metric, covariant and contravariant indices, automatic index raising/lowering, Einstein summation via repeated indices, symmetry properties of indices, and canonicalization of tensor expressions. There's also a `toperators` submodule for things like partial-derivative operators on tensors. This is what you'd reach for if you want genuine index gymnastics rather than just arrays of expressions.

**Physics and differential geometry.** Beyond the core tensor module, `sympy.diffgeom` handles manifolds, coordinate systems, and differential forms, and the physics submodules include vector/dyadic machinery (`sympy.physics.vector`) and mechanics tooling. For general relativity specifically, people often compute Christoffel symbols, the Riemann and Ricci tensors, etc., by combining the N-dim arrays with `derive_by_array` and contractions — and there are well-known third-party libraries (like EinsteinPy and GraviPy) built on top of SymPy that package this more conveniently.

A couple of honest caveats. The different modules don't fully interoperate — the array module, the indexed module, and the abstract tensor module each have their own conventions, so you generally pick the one that fits your task rather than mixing them. The abstract `tensor` module has a steeper learning curve and somewhat terse documentation. And SymPy tensors are symbolic, so they're excellent for derivation and exact results but slow for large numerical work, where NumPy or specialized libraries are more appropriate.

For teaching purposes, the N-dimensional `Array` plus `tensorproduct`/`tensorcontraction`/`derive_by_array` is usually the sweet spot: it's intuitive, stays symbolic and exact, and lets students see tensor products and contractions concretely before introducing the heavier abstract-index machinery. If you let me know whether you're aiming at physics/GR or a more pure multilinear-algebra treatment, I can point you to the most suitable module and put together some worked examples.Yes, SymPy has fairly substantial tensor support, spread across a few different modules that serve different purposes. The tensor functionality is organized into N-dimensional arrays, indexed objects, and a full abstract tensor module, plus additional differential-geometry and physics tooling.

**N-dimensional arrays (`sympy.tensor.array`).** This is the most accessible entry point and the closest analog to a NumPy array but with symbolic entries. You get dense and sparse, mutable and immutable variants, and the usual operations: tensor product, contraction, element-wise application, derivatives, and reshaping.

```python
from sympy import symbols
from sympy.tensor.array import Array, tensorproduct, tensorcontraction, derive_by_array

x, y, z = symbols('x y z')
A = Array([[x, y], [z, x*y]])      # a 2x2 symbolic array
B = Array([1, 2])
tensorproduct(A, B)                # outer product -> higher-rank array
tensorcontraction(A, (0, 1))       # contract axes 0 and 1 (the trace here)
derive_by_array([x*y, x+z], [x, y, z])  # Jacobian-style differentiation
```

This is great for symbolic multilinear algebra and for differentiating expressions with respect to vectors/matrices.

**Indexed objects (`sympy.tensor.indexed`).** This module provides `IndexedBase`, `Indexed`, and `Idx`, which let you write things in abstract index notation like `A[i, j]*x[j]` without committing to concrete values. It's primarily symbolic bookkeeping — useful for deriving expressions and for code generation — but it does not by itself enforce Einstein summation or tensor transformation rules.

**Abstract tensors (`sympy.tensor.tensor`).** This is the most "real" tensor module in the differential-geometry/physics sense. It supports tensors defined over `TensorIndexType`s with a metric, covariant and contravariant indices, automatic index raising/lowering, Einstein summation via repeated indices, symmetry properties of indices, and canonicalization of tensor expressions. There's also a `toperators` submodule for things like partial-derivative operators on tensors. This is what you'd reach for if you want genuine index gymnastics rather than just arrays of expressions.

**Physics and differential geometry.** Beyond the core tensor module, `sympy.diffgeom` handles manifolds, coordinate systems, and differential forms, and the physics submodules include vector/dyadic machinery (`sympy.physics.vector`) and mechanics tooling. For general relativity specifically, people often compute Christoffel symbols, the Riemann and Ricci tensors, etc., by combining the N-dim arrays with `derive_by_array` and contractions — and there are well-known third-party libraries (like EinsteinPy and GraviPy) built on top of SymPy that package this more conveniently.

A couple of honest caveats. The different modules don't fully interoperate — the array module, the indexed module, and the abstract tensor module each have their own conventions, so you generally pick the one that fits your task rather than mixing them. The abstract `tensor` module has a steeper learning curve and somewhat terse documentation. And SymPy tensors are symbolic, so they're excellent for derivation and exact results but slow for large numerical work, where NumPy or specialized libraries are more appropriate.

For teaching purposes, the N-dimensional `Array` plus `tensorproduct`/`tensorcontraction`/`derive_by_array` is usually the sweet spot: it's intuitive, stays symbolic and exact, and lets students see tensor products and contractions concretely before introducing the heavier abstract-index machinery. If you let me know whether you're aiming at physics/GR or a more pure multilinear-algebra treatment, I can point you to the most suitable module and put together some worked examples.Yes, SymPy has fairly substantial tensor support, spread across a few different modules that serve different purposes. The tensor functionality is organized into N-dimensional arrays, indexed objects, and a full abstract tensor module, plus additional differential-geometry and physics tooling.

**N-dimensional arrays (`sympy.tensor.array`).** This is the most accessible entry point and the closest analog to a NumPy array but with symbolic entries. You get dense and sparse, mutable and immutable variants, and the usual operations: tensor product, contraction, element-wise application, derivatives, and reshaping.

```python
from sympy import symbols
from sympy.tensor.array import Array, tensorproduct, tensorcontraction, derive_by_array

x, y, z = symbols('x y z')
A = Array([[x, y], [z, x*y]])      # a 2x2 symbolic array
B = Array([1, 2])
tensorproduct(A, B)                # outer product -> higher-rank array
tensorcontraction(A, (0, 1))       # contract axes 0 and 1 (the trace here)
derive_by_array([x*y, x+z], [x, y, z])  # Jacobian-style differentiation
```

This is great for symbolic multilinear algebra and for differentiating expressions with respect to vectors/matrices.

**Indexed objects (`sympy.tensor.indexed`).** This module provides `IndexedBase`, `Indexed`, and `Idx`, which let you write things in abstract index notation like `A[i, j]*x[j]` without committing to concrete values. It's primarily symbolic bookkeeping — useful for deriving expressions and for code generation — but it does not by itself enforce Einstein summation or tensor transformation rules.

**Abstract tensors (`sympy.tensor.tensor`).** This is the most "real" tensor module in the differential-geometry/physics sense. It supports tensors defined over `TensorIndexType`s with a metric, covariant and contravariant indices, automatic index raising/lowering, Einstein summation via repeated indices, symmetry properties of indices, and canonicalization of tensor expressions. There's also a `toperators` submodule for things like partial-derivative operators on tensors. This is what you'd reach for if you want genuine index gymnastics rather than just arrays of expressions.

**Physics and differential geometry.** Beyond the core tensor module, `sympy.diffgeom` handles manifolds, coordinate systems, and differential forms, and the physics submodules include vector/dyadic machinery (`sympy.physics.vector`) and mechanics tooling. For general relativity specifically, people often compute Christoffel symbols, the Riemann and Ricci tensors, etc., by combining the N-dim arrays with `derive_by_array` and contractions — and there are well-known third-party libraries (like EinsteinPy and GraviPy) built on top of SymPy that package this more conveniently.

A couple of honest caveats. The different modules don't fully interoperate — the array module, the indexed module, and the abstract tensor module each have their own conventions, so you generally pick the one that fits your task rather than mixing them. The abstract `tensor` module has a steeper learning curve and somewhat terse documentation. And SymPy tensors are symbolic, so they're excellent for derivation and exact results but slow for large numerical work, where NumPy or specialized libraries are more appropriate.

For teaching purposes, the N-dimensional `Array` plus `tensorproduct`/`tensorcontraction`/`derive_by_array` is usually the sweet spot: it's intuitive, stays symbolic and exact, and lets students see tensor products and contractions concretely before introducing the heavier abstract-index machinery. If you let me know whether you're aiming at physics/GR or a more pure multilinear-algebra treatment, I can point you to the most suitable module and put together some worked examples.GR or a more pure multilinear-algebra treatment, I can point you to the most suitable module and put together some worked examples.
