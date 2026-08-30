# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: doingmath (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # kingdon geometric algebra library
# summary notebook based upon the documentation, a lot of copy/paste
# %%
import numpy as np
import kingdon  # as kd?
from kingdon import Algebra

alg2 = Algebra(p=2, q=0, r=0)  # use 3D VGA for simple examples
alg3 = Algebra(p=3, q=0, r=0)  # use 3D VGA for simple examples
locals().update(alg3.blades)  # make VGA 3 the default but ok to use other algebras
# %% [markdown]
# ### Symbolic Multivectors
# In order to create symbolical multivectors in an algebra, we can call multivector and explicitly pass a name argument. For example, let us create two symbolic vectors u and v in this algebra
# %%
u = alg3.multivector(name="u", grades=(1,))
v = alg3.multivector(name="v", grades=(1,))
u, v
# %% [markdown]
# this is an example for 2D VGA - alittle different from above
# ```python
# v = alg2.multivector(name='v', grades=(1,))
# u
# u1 𝐞₁ + u2 𝐞₂
# v
# v1 𝐞₁ + v2 𝐞₂
# ```
# %% [markdown]
# ### common operations, addition, (geometric) product
# ```
# >>> u + v # (multi) vector addition
# (u1 + v1) 𝐞₁ + (u2 + v2) 𝐞₂
# >>> u * v # geometric product
# (u1*v1 + u2*v2) + (u1*v2 - u2*v1) 𝐞₁₂
# ```
# %%
u + v
# %%
u * v
# %%
(u + v) ** 2
# (u1**2 + 2*u1*v1 + u2**2 + 2*u2*v2 + u3**2 + 2*u3*v3 + v1**2 + v2**2 + v3**2)
# %% [markdown]
# ### inner (dot) and exterior (wedge) products
# ```
# u | v
# (u1*v1 + u2*v2)
# u ^ v
# (u1*v2 - u2*v1) 𝐞₁₂
# ```
# %%
u | v  # dot

# for 2D: (u1*v1 + u2*v2)
# %%
u ^ v  # wedge
# for 2D: (u1*v2 - u2*v1) 𝐞₁₂
# %%
(u + v) * (u + v)
# (u1**2 + 2*u1*v1 + u2**2 + 2*u2*v2 + u3**2 + 2*u3*v3 + v1**2 + v2**2 + v3**2)
# %%
(u ^ v) ** 2  # -> scalar
# (-u1**2*v2**2 - u1**2*v3**2 + 2*u1*u2*v1*v2 + 2*u1*u3*v1*v3 - u2**2*v1**2 - u2**2*v3**2 + 2*u2*u3*v2*v3 - u3**2*v1**2 - u3**2*v2**2)


# %% [markdown]
# ### Conjugation
# temporarily switch back to 2D VGA
# u = alg2.multivector(name='u', grades=(1,))
# v = alg2.multivector(name='v', grades=(1,))
# u,v
# %%
# conjugation in 2D VGA, conjugation by a vector u <-> reflection across the line defined by u
u >> v
# (-u1**2*v1 - 2*u1*u2*v2 + u2**2*v1) 𝐞₁ + (u1**2*v2 - 2*u1*u2*v1 - u2**2*v2) 𝐞₂
# %%
# compare with its longer version which uses negative reverse instead of inverse
u * v * (-u.reverse())

# %%
u >> v == u * v * (-u.reverse())

# %%
# compare with true conjugation
u * v * u.inv()

# %%
# check if the assumptions of the operator which uses the reverse instead of the inverse
~u.normalized() * u.normalized()

# %%

# %% [markdown]
# # using sympy
# ```
# from sympy import Symbol, sin, cos
# t = Symbol('t')
# x = cos(t) * e + sin(t) * e12
# x.normsq()
# 1
# ```
#
# Or. Strings are automatically converted to symbolics with sympy. so x from the
# example above can also be created as:
#
# ```
# x  = alg.multivector(e='cos(t)', e12='sin(t)')
# x.normsq()
# 1
# ```
# ### More control over basisvectors
#
# If we do not just want to create a symbolic multivector of a certain grade,
# but with specific blades, we can do so by providing the keys argument.
# ```
# >>> x = alg.multivector(name='x', keys=('e1', 'e12'))
# >>> x1 𝐞₁ + x12 𝐞₁₂
# ```
#
# This can be done either by providing a tuple of strings which indicate which
# basis-vectors should be present, or by passing them as integers,
# i.e. keys=(0b01, 0b11) is equivalent to the example above. Internally, kingdon
# uses the binary representation.
#
# %% [markdown]
# ## Numerical Multivectors
#
# While `kingdon` makes no assumptoins about the data structures that are passed
# into a multivector in order to support ducktyping and customization as much as
# possible, it was nonetheless designed to work really well with numpy arrays.
#
# For example, to repeat some of the examples above with numerical values, we could do
#
# ```
# import numpy as np
# uvals, vvals = np.random.random((2, 2))
# u = alg.vector(uvals)
# v = alg.vector(vvals)
# u * v
# (0.1541) + (0.0886) 𝐞₁₂
# ```
#
# %%
import numpy as np

uvals, vvals = np.random.random((2, 2))
u = alg2.vector(uvals)
v = alg2.vector(vvals)
u * v

# %% [markdown]
#
# A big performance bottleneck that we suffer from in Python, is that arrays over objects are very slow. So while we could make a numpy array filled with ~kingdon.multivector.MultiVector’s, this would tank our performance. kingdon gets around this problem by instead accepting numpy arrays as input. So to make a collection of 3 lines, we do
# ```python
# import numpy as np
# uvals = np.random.random((2, 3))
# u = alg.vector(uvals)
# u
# ([0.82499172 0.71181276 0.98052928]) 𝐞₁ + ([0.53395072 0.07312351 0.42464341]) 𝐞₂
#

# %%
import numpy as np

uvals = np.random.random((2, 3))
u = alg2.vector(uvals)
u
# ([0.82499172 0.71181276 0.98052928]) 𝐞₁ + ([0.53395072 0.07312351 0.42464341]) 𝐞₂

# %%
u**2  # returns a arrayified-grade0 multivector

# %% [markdown]
# ```
#
# what is important here is that the first dimension of the array has to have the expected length: 2 for a vector. All other dimensions are not used by kingdon. Now we can reflect this multivector in the e1 line:
# ```python
# v = alg.vector((1, 0))
# v >> u
#
# ([0.82499172 0.71181276 0.98052928]) 𝐞₁ + ([-0.53395072 -0.07312351 -0.42464341]) 𝐞₂
# ```
#

# %%
v = alg2.vector((1, 0))
v >> u


# %% [markdown]
#
# Despite the different shapes, broadcasting is done correctly in the background thanks to the magic of numpy, and with only minor performance penalties.
#
#

# %% [markdown]
# ### The Shape of MultiVectors
# The first dimension of a multivector is always the coefficients of the multivector. For example, to create a vector in ℝ3 we could do

# %%
from kingdon import Algebra
import numpy as np

alg3 = Algebra(3)
xvals = np.random.rand(3)
x = alg3.vector(xvals)
x.shape
# (3,)

# %% [markdown]
# Now if we look at x.shape, we see that it is (3,), the same as xvals.shape. However, the length of x is 0:

# %%
len(x)

# %% [markdown]
# This reflects that x is a single vector, and therefore not iterable. You might have expected iteration over a multivector to iterate over its coefficients, but in kingdon multivectors are treated as geometric numbers, similar to how complex numbers are treated in complex analysis.

# %% [markdown]
# Note
# If you need to iterate over the coefficients anyway use x.map to map a function on all the coefficients of the multivector. For individual access, use attributes access instead, e.g. x.e1 returns the 𝐞1 coefficient.

# %%
x.e, x.e1, x.e2

# %% [markdown]
# Now lets make a collection of 𝑁 vectors, and see what changes:
# ```python
# N = 5
# xvals = np.random.rand(3, N)
# x = alg.vector(xvals)
# x
# [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864] 𝐞₁ + [0.15599452 0.05808361 0.86617615 0.60111501 0.70807258] 𝐞₂ + [0.02058449 0.96990985 0.83244264 0.21233911 0.18182497] 𝐞₃
# x.shape
# (3, 5)
# len(x)
# 5
# ```

# %% [markdown]
# Hence, we see that the length of the multivector is 5, and therefore we can iterate over the multivector to get the individual vectors in x:

# %%

# %% [markdown]
#
# ### Operators
#
# |Operation        |Expression |Infix |Inline |
# |-----------------|-----------|------|-------|
# |Geometric product|$𝑎⁢𝑏$       |a*b   |a.gp(b)|
# |Inner |𝑎 ⋅ 𝑏 |a\|b |a.ip(b) |
# |Scalar product |$⟨𝑎⋅𝑏⟩_0$|(a\|b).grade(0)|a.sp(b) |
# |Left-contraction | $𝑎⌋𝑏$ | |a.lc(b) |
# |Right-contraction| $𝑎⌊𝑏$ | |a.rc(b) |
# |Outer (Exterior) | 𝑎 ∧ 𝑏 | a ^ b|a.op(b) |
# |Regressive | 𝑎 ∨ 𝑏 | a & b |a.rp(b) |
# |Conjugate a by b|$(−1)^{\rm grade⁡(𝑏)⁢grade⁡(𝑎)⁢} 𝑏⁢𝑎⁢\widetilde{b}$| b >> a| b.sw(a) $\widetilde{b} b =1$|
# |Project a onto b | (𝑎 ⋅ 𝑏) 𝑏̃        |a @ b          | a.proj(b)    |
# |Commutator of a and b | 𝑎×𝑏 = $\frac{⁢[𝑎,𝑏]}{2}$|  | a.cp(b)      |
# |Anti-commutator of a and b| ⁢{𝑎,𝑏}/2     |    | a.acp(b)     |
# |Sum of a and b   | 𝑎+𝑏            | a + b         | a.add(b)     |
# |Difference of a and b| 𝑎 −𝑏       | a - b         | a.sub(b)     |
# |Reverse of a     | 𝑎̃              | ~a            | a.reverse()  |
# |Squared norm of a| 𝑎𝑎̃             |               | a.normsq()   |
# |Norm of a        | √𝑎𝑎̃            |               | a.norm()     |
# |Normalize a      | 𝑎/√𝑎𝑎̃          |               | a.normalized()|
# |Square root of a | √𝑎             |               | a.sqrt()     |
# |Hodge Dual of a  | $\star 𝑎$      |               | a.dual()     |
# |Undual of a      |                |               | a.undual()   |
# |Grade k part of a| $⟨𝑎⟩_𝑘$        |               |a.grade(k)    |
#
# Note that formally conjugation is defined by $𝑏⁢𝑎⁢𝑏^{−1}$ and
# projection by $(𝑎 ⋅𝑏)⁢𝑏^{−1}$, but that both are implemented using reversion
# instead of an inverse. This is because reversion is much faster to calculate,
# and because in practice 𝑏 will often by either a rotor satisfying $ 𝑏⁢𝑏̃ =1$ or
# a blade satisfying $𝑏^2 = 𝑏⋅𝑏$, and thus the inverse is identical to the
# reverse (up to sign).
#
# If you want to replace these operators by their proper definitions, you can
# use the register decorator to overwrite the default operator (use at your own
# risk):
# ```python
# @alg.register(name='sw')
# def sw(x, y):
#     return x * y / y
#
# @alg.register(name='proj')
# def proj(x, y):
#     return (x | y) / y
# ```

# %% [markdown]
# ## Graphing
# - see separate notebook
# %% [markdown]
# ## Large Algebra's
# In theory kingdon supports algebra’s up to 36D, but your computer might go up in smoke if you push it that far. In order to make large’s algebras feasible, kingdon no longer performs symbolic optimization and caching because this consumes to much memory, and instead just computes naively. By default any algebra of 𝑑 >6 is considered large, but it can be forced manually with the large option to Algebra depending on your needs:
# ```
# alg = Algebra(3, large=True)
# alg = Algebra(8, large=False)
# ```
# ## Performance Tips
# Because kingdon attempts to symbolically optimize expressions the first time they are called, the first call to any operation is comparatively slow, whereas subsequent calls have very good performance.
#
# There are however several things to be aware of to ensure good performance.
#
# ### Broadcasting
# Avoid arrays of multivectors, and use multivectors over e.g. numpy arrays or PyTorch tensors instead, as shown in the numerical section. This ensures the high level overhead of kingdon is paid only once.
#
# ### Register Expressions
# To make it easy to optimize larger expressions, kingdon offers the register() decorator.
# ```python
# alg = Algebra(3, 0, 1)
#
# @alg.register
# def myfunc(u, v):
#      return u * (u + v)
#
# x = alg.vector(np.random.random(4))
# y = alg.vector(np.random.random(4))
# myfunc(x, y)
# ```
#
# Calling the decorated myfunc has the benefit that all the numerical computation is done in one single call, instead of doing each binary operation individually. This has the benefit that all the (expensive) python boilerplate code is called only once. Moreover, one can use @alg.register(symbolic=True)
#
# ### Graded
# The first time kingdon is asked to perform an operation it hasn’t seen before, it performs code generation for that particular request. Because codegen is the most expensive step, it can be beneficial to reduce the number of times it is needed. An easy way to achieve this is to initiate the Algebra with graded=True. This enforces that kingdon does not specialize codegen down to the individual basis blades, but rather only per grade. This means there are far less combinations that have to be considered and generated.
#
# ### Numba JIT
# We can enable numba just-in-time compilation by initiating an Algebra with wrapper=numba.njit. This comes with a significant cost the first time any operator is called, but subsequent calls to the same operator are significantly faster. It is worth mentioning that when dealing with Numerical Multivectors over numpy arrays, the benefit of using numba actually disappears rapidly as the numpy arrays become larger, since then most of the time is spend in numpy routines anyway. So you need to experiment carefully if numba is right for you.

# %% [markdown]
#

# %%
e1

# %%
e1.dual()

# %%
e1.dual().dual()

# %%
e1.dual().undual()

# %%
# cross product u x v
(u ^ v).dual()


# %%
def toscalar(mvec: kingdon.MultiVector):
    """take the grade0 "scalar" portion of a multi vector into a regular float
    kingdon.MultiVector <-> kingdon.multivector.MultiVector

    it threads over multivectors that are composed with arrays
    """
    return mvec.e
    g0 = mvec.grade(0)
    vals = g0.values()
    # use g0.map instead ?
    # use g0.e instead?
    if vals:
        r = vals[0]
    else:
        r = 0
    return r


# kingdon.MultiVector


# %%
epsilon = 10 ** (-4)
print("basic [ok]" if toscalar(x**2) - 0.415 < epsilon else "error")
print(
    "threaded [ok]"
    if np.allclose(toscalar(u**2), np.array([0.51851588, 0.14950576, 0.21833548]))
    else "error!"
)


# %% [markdown]
# ### Mapping `toscalar` over arbitrarily nested containers
#
# The subtlety is *not* the recursion, it is deciding what counts as a
# container to descend into vs. a leaf to apply the function to.
# `kingdon.MultiVector` defines `__getitem__` but no `__iter__`, so it is **not**
# an `abc.Iterable`, yet `iter(mv)` still succeeds via the legacy sequence
# protocol and yields *nothing* — a duck-typed `try: iter(x)` test would quietly
# turn every multivector into `[]`. So test against the ABC, not `iter()`.

# %%
from collections.abc import Iterable

# containers we descend into; everything else is a leaf
ATOMIC = (str, bytes, bytearray, kingdon.MultiVector)


def is_container(obj) -> bool:
    return isinstance(obj, Iterable) and not isinstance(obj, ATOMIC)


def _rebuild(obj, items):
    """Repackage `items` as the same kind of container as `obj`."""
    if isinstance(obj, np.ndarray):
        # NB: `np.ndarray(items)` is the raw uninitialised-memory constructor
        # (first arg is a *shape*), not a from-sequence factory. It does not
        # raise, it returns garbage -- so ndarray must be special-cased.
        return np.array(items)
    if isinstance(obj, (tuple, set, frozenset)):
        try:
            return type(obj)(items)
        except TypeError:
            return type(obj)(*items)  # namedtuples take positional args
    return list(items)  # lists, generators, ranges, dict_values


def deep_map(func, obj):
    """Apply `func` to every leaf of an arbitrarily nested iterable,
    preserving the nesting structure and (where possible) the container type.

    >>> deep_map(toscalar, [[e1 | e1, e1 | e2], [e2 | e1, e2 | e2]])
    [[1, 0], [0, 1]]
    """
    if not is_container(obj):
        return func(obj)
    # materialise first: rebuilding must not half-consume a lazy iterator
    return _rebuild(obj, [deep_map(func, x) for x in obj])


def deep_toscalar(obj):
    return deep_map(toscalar, obj)


# %% [markdown]
#
