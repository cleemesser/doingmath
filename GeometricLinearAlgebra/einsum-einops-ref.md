---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Einsum and Einops — a working reference

A single runnable notebook covering the Einstein summation convention as implmented in python. The major numeric array packages (numpy, jax, pytorch,tensorflow)  support such operations via native implementations. [Einops](https://einops.rocks/) extends this to allow multi-letter index names and provides operations beyond summation: `rearrange`, `reduce`, `repeat`, `pack`, `unpack`. 

Each section pairs the mathematical defintion of the operation with runnable code that demonstrates the actions. 
operation actually does. Run top to bottom; outputs are deterministic so
you can check them against the math.

At the end there is a multi-head attention example built from `rearrange` and `jnp.einsum`.


+++

## 0. `einsum` primer — the common operations

The Einstein Summation Convention expresses tensor operations by naming axes.

```{important}
**The one rule of `einsum`:** a repeated index that is absent from the output
is summed over; every other index is carried through.
```

Here are its everyday uses.

```{code-cell} ipython3
# initial imports
import einops
import numpy as np
try:
  import jax.numpy as jnp
except ImportError:
  jnp = None

import sympy as sp  # sympy for nice display
sp.init_printing()
```

```{code-cell} ipython3
# display helpers
from IPython.display import Math

def show_eq(name, x):
    """LaTeX `name = [x]`, for use in a MyST {eval} role."""
    x = np.asarray(x)
    shape = r" \times ".join(map(str, x.shape))
    return Math(rf"{name} = {sp.latex(sp.Matrix(x))}")

def show_inR(name, x):
    """LaTeX `name = [x] in R^{shape}`, for use in a MyST {eval} role."""
    x = np.asarray(x)
    shape = r" \times ".join(map(str, x.shape))
    return Math(rf"{name} = {sp.latex(sp.Matrix(x))} \in \mathbb{{R}}^{{{shape}}}")

# experimenting
def str2md(s):
    return s.replace('\n', '<br>') # nope
```

Let's create example arrays that are small enough to write out in full. We will use them in examples throughout section 0.

+++

In code these are `a`, `b`, and `v`:

{eval}`show_inR("A", a)` &emsp; {eval}`show_inR("B", b)` &emsp; {eval}`show_inR("v", v)`

### 0a. Matrix multiply: `'ik,kj->ij'`

$k$ is repeated and dropped, so it is summed. In classical mathematical notation a matrix multiplication is written as:

```{math}
:label: eq-matmul
C_{ij} = \sum_{k} A_{ik} B_{kj}.
```

If the Einstein summation convention is used in mathematics, then the above is simply written as:

```{math}
:label: eq-matmul-ein-convention
C_{ij} = A_{ik} B_{kj}.
```
Note the repeated $k$ index which has the same range in both A and B.

Each entry of $C$ is row $i$ of $A$ dotted with column $j$ of $B$. For
example, $C_{00} = 0\cdot 0 + 1\cdot 4 + 2\cdot 8 = 20$. The full product is

$$
\underbrace{\begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \end{bmatrix}}_{A\;(2\times 3)}
\underbrace{\begin{bmatrix}
0 & 1 & 2 & 3 \\
4 & 5 & 6 & 7 \\
8 & 9 & 10 & 11
\end{bmatrix}}_{B\;(3\times 4)}
=
\underbrace{\begin{bmatrix}
20 & 23 & 26 & 29 \\
56 & 68 & 80 & 92
\end{bmatrix}}_{C\;(2\times 4)}.
$$

```{code-cell} ipython3
# define a, b, calculdate their matrix multiplication
a = np.arange(6).reshape(2,3) # 2 x 3
b = np.arange(12).reshape(3,4) # 3 x 4
jnp.einsum("ik,kj->ij", a, b)  # (2, 4)
```

### 0b. Transpose: `'ij->ji'`

No index is dropped — just a reordering:

$$
C_{ji} = A_{ij},
\qquad
\begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \end{bmatrix}^{\mathsf T}
=
\begin{bmatrix} 0 & 3 \\ 1 & 4 \\ 2 & 5 \end{bmatrix}.
$$

```{code-cell} ipython3
jnp.einsum("ij->ji", a)  # (3, 2)
```

### 0c. Sum all / along an axis

Dropping every index sums the whole tensor; dropping one axis sums along it:

$$
s = \sum_{i}\sum_{j} A_{ij},
\qquad
r_{j} = \sum_{i} A_{ij}.
$$

For our $A$, $s = 0+1+2+3+4+5 = 15$, and summing down each column gives

$$
r = \begin{bmatrix} 0+3 & 1+4 & 2+5 \end{bmatrix}
  = \begin{bmatrix} 3 & 5 & 7 \end{bmatrix}.
$$

```{code-cell} ipython3
print("a=\n", a)
print(jnp.einsum("ij->", a))  # scalar: sum of all
print(jnp.einsum("ij->j", a))  # (3,): sum over rows (down each column)
```

### 0d. Matrix–vector product: `'ij,j->i'`

$$
y_{i} = \sum_{j} A_{ij}\, x_{j},
\qquad
\begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \end{bmatrix}
\begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}
=
\begin{bmatrix} 0\cdot 0 + 1\cdot 1 + 2\cdot 2 \\ 3\cdot 0 + 4\cdot 1 + 5\cdot 2 \end{bmatrix}
=
\begin{bmatrix} 5 \\ 14 \end{bmatrix}.
$$

This is {eq}`eq-matmul` with $B$ replaced by a single column.

```{code-cell} ipython3
jnp.einsum("ij,j->i", a, v)  # (2,)
```

### 0e. Dot product and outer product

Dot contracts the shared index; outer keeps both (nothing summed):

$$
d = \sum_{i} u_i w_i,
\qquad
O_{ij} = u_i w_j.
$$

With $u = (1, 2, 3)$ and $w = (4, 5, 6)$:

$$
d = 1\cdot 4 + 2\cdot 5 + 3\cdot 6 = 32,
\qquad
O = u\,w^{\mathsf T}
= \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
  \begin{bmatrix} 4 & 5 & 6 \end{bmatrix}
= \begin{bmatrix}
4 & 5 & 6 \\
8 & 10 & 12 \\
12 & 15 & 18
\end{bmatrix}.
$$

```{code-cell} ipython3
u = np.array([1.0, 2.0, 3.0])
w = np.array([4.0, 5.0, 6.0])
```

{eval}` show_eq("u",u)`
{eval}` show_eq("w", w)`

w = {eval}`w`

$u^{\mathsf T}\,w = $ {eval}`jnp.einsum("i,i->", u,w)`

$u\,w^{\mathsf T} = $ {eval}`f"{jnp.einsum("i,j->ij", u,w)}"`

$u\,w^{\mathsf T} = $ {eval}`str2md(f"{jnp.einsum("i,j->ij", u,w)}")`

{eval}`show_eq(r"u\,w^{\mathsf T}", jnp.einsum("i,j->ij", u,w))`

+++

### 0f. Trace and batched matmul

Trace repeats an index on a single input; batched matmul carries a batch
axis `b` through untouched while contracting `k`:

$$
t = \sum_{i} M_{ii},
\qquad
C_{b,i,j} = \sum_{k} A_{b,i,k} B_{b,k,j}.
$$

For the trace example, only the diagonal survives:

$$
\operatorname{tr}\begin{bmatrix}
\mathbf{0} & 1 & 2 \\
3 & \mathbf{4} & 5 \\
6 & 7 & \mathbf{8}
\end{bmatrix} = 0 + 4 + 8 = 12.
$$

Batched matmul is {eq}`eq-matmul` applied independently to each of the 5
slices along $b$: five $(2\times 3)(3\times 4)$ products stacked into a
$(5, 2, 4)$ result.

```{code-cell} ipython3
M = jnp.arange(9).reshape(3, 3)
print("trace:", jnp.einsum("ii->", M))

A = jnp.ones((5, 2, 3))
B = jnp.ones((5, 3, 4))
print("batched matmul:", jnp.einsum("bik,bkj->bij", A, B).shape)  # (5, 2, 4)
```

### 0g. Batched matmul in detail: `'bik,bkj->bij'`

The all-ones example above only checks shapes, since every product is the
same. To see what happens to each batch, use a batch of two $2\times 2$
matrices whose products are easy to tell apart.

**Read the subscripts one index at a time.** Apply the one rule to each letter:

| index | in `A`? | in `B`? | in output? | role |
|:-----:|:-------:|:-------:|:----------:|------|
| `b` | yes | yes | yes | **batch**: carried through, so slice $b$ of $A$ only meets slice $b$ of $B$ |
| `i` | yes | no | yes | kept: the rows of each $A_b$ |
| `j` | no | yes | yes | kept: the columns of each $B_b$ |
| `k` | yes | yes | no | **summed**: the inner dimension of each product |

Note the difference between `b` and `k`. Both appear in both inputs, but only
`k` is dropped from the output, so only `k` is summed. Keeping `b` in the
output is what makes the batches stay separate.

**Fix $b$ and you get an ordinary matrix product.** Write $A_b$ for the
$2\times 2$ matrix $A_{b,:,:}$. For each fixed $b$, the formula
$C_{b,i,j} = \sum_k A_{b,i,k} B_{b,k,j}$ is exactly {eq}`eq-matmul`:

```{math}
:label: eq-batched
C_b = A_b B_b, \qquad b = 0, 1, \dots
```

**A worked example.** Let

$$
A_0 = \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix},
\quad
A_1 = \begin{bmatrix} 4 & 5 \\ 6 & 7 \end{bmatrix},
\qquad
B_0 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},
\quad
B_1 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

$B_0$ is the identity and $B_1$ swaps the two columns of whatever it
multiplies on the right. So each batch gets a different operation:

$$
C_0 = A_0 B_0
= \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix}
  \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
= \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix},
\qquad
C_1 = A_1 B_1
= \begin{bmatrix} 4 & 5 \\ 6 & 7 \end{bmatrix}
  \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
= \begin{bmatrix} 5 & 4 \\ 7 & 6 \end{bmatrix}.
$$

One entry written out in full, with $b = 1$, $i = 0$, $j = 0$:

$$
C_{1,0,0} = \sum_{k=0}^{1} A_{1,0,k}\, B_{1,k,0}
= A_{1,0,0} B_{1,0,0} + A_{1,0,1} B_{1,1,0}
= 4\cdot 0 + 5\cdot 1 = 5.
$$

The result stacks $C_0$ and $C_1$ along the batch axis, giving shape
$(2, 2, 2)$. `A_0` never meets `B_1`: no term in the sum mixes batches.

```{code-cell} ipython3
Ab = jnp.arange(8.0).reshape(2, 2, 2)  # A_0, A_1
Bb = jnp.array([[[1.0, 0.0], [0.0, 1.0]],  # B_0: identity
                [[0.0, 1.0], [1.0, 0.0]]])  # B_1: swap columns
Cb = jnp.einsum("bik,bkj->bij", Ab, Bb)
print("C_0 =\n", Cb[0])
print("C_1 =\n", Cb[1])

# the same thing three ways: a Python loop over b, and the @ operator
C_loop = jnp.stack([Ab[n] @ Bb[n] for n in range(Ab.shape[0])])
print("matches loop of A_b @ B_b:", np.allclose(Cb, C_loop))
print("matches Ab @ Bb:          ", np.allclose(Cb, Ab @ Bb))
```

**Two variations** show how each letter matters.

If $B$ has no batch axis, drop `b` from its subscript: `'bik,kj->bij'`.
Now one matrix $B$ multiplies every $A_b$, so $C_b = A_b B$. This is
an example of broadcasting. The einsum notation makes explicit.

If you also drop `b` from the output, `'bik,bkj->ij'`, then `b` becomes a
repeated index that is absent from the output, so it is summed too. You get
the sum of the batch products, not a stack of them:

$$
\sum_b A_b B_b = C_0 + C_1
= \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix}
+ \begin{bmatrix} 5 & 4 \\ 7 & 6 \end{bmatrix}
= \begin{bmatrix} 5 & 5 \\ 9 & 9 \end{bmatrix}.
$$

```{code-cell} ipython3
print("shared B (B_1 for every batch):\n", jnp.einsum("bik,kj->bij", Ab, Bb[1]))
print("batch summed away, shape", jnp.einsum("bik,bkj->ij", Ab, Bb).shape, ":\n",
      jnp.einsum("bik,bkj->ij", Ab, Bb))
```

**Hadamard Product: elementwise multiplication**
The elementwise, Hadamard product, often written $C = A \odot B$. 
$$
C_{ij} = A_{ij} B_{ij}
$$
Note that the presence of the i,j indices on the left means we ignore the repeated indices on the right.

Note that this is also the same as the usual numpy multiplication operator for 1 dimensional arrays.

```{code-cell} ipython3
A = np.arange(6).reshape(2,3)
B = np.arange(6,12).reshape(2,3)
C = np.einsum("i j, i j->i j",A,B) # spaces not required but allowed
# note einops.einsum has slightly different conventions
# it uses spaces to separate indices because it allows multi-character
# index names
# out = einsum(x, y, z, "a b c, c b d, a g k -> a b k")
C2 = einops.einsum(A,B,"i j,i j->i j") 
assert np.all(C == C2)
C
```

Note that this is the same as the usual numpy multiplcation operator between two arrays, `u * w`:

`u` = {eval}`u`, `w` = {eval}`w`

`u * v` :=> {eval}`u * w`

`np.einsum("i,i->i",u,w)` :=>
{eval}`np.einsum("i,i->i",u,w)`

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax import nn
from einops import rearrange, reduce, repeat

print("jax", jax.__version__)
```

## 1. `rearrange` — reshape and reorder axes

`rearrange` permutes, splits, and merges axes. It never changes the data,
only how it is laid out. The pattern string names the input axes on the
left of `->` and the desired layout on the right.

```{code-cell} ipython3
x = jnp.arange(24).reshape(2, 3, 4)
print("x shape:", x.shape)
x
```

### 1a. Transpose: `'a b c -> a c b'`

Swapping the last two axes is an index permutation. If
$X \in \mathbb{R}^{A\times B\times C}$ and $Y \in \mathbb{R}^{A\times C\times B}$,

$$
Y_{a,c,b} = X_{a,b,c}.
$$

```{code-cell} ipython3
y = rearrange(x, "a b c -> a c b")
print("shape:", y.shape)  # (2, 4, 3)
y
```

### 1b. Flatten: `'a b c -> a (b c)'`

Merging axes $b$ and $c$ into one axis of size $BC$ with row-major
(C-order) indexing:

```{math}
:label: eq-merge
Y_{a,\; bC + c} = X_{a,b,c},
\qquad 0 \le b < B,\; 0 \le c < C.
```

```{code-cell} ipython3
y = rearrange(x, "a b c -> a (b c)")
print("shape:", y.shape)  # (2, 12)
y
```

### 1c. Split: `'(h2 h) w -> h2 h w'`

Splitting an axis of size $H$ into two axes $H_2$ and $H'$ with
$H = H_2 \cdot H'$. The inverse of the merge {eq}`eq-merge`:

$$
Y_{h_2,\,h',\,w} = X_{\,h_2 H' + h',\; w}.
$$

You must supply one of the two sizes (here `h2`) so the split is determined.

```{code-cell} ipython3
img = jnp.arange(24).reshape(4, 6)
print("img shape:", img.shape)
y = rearrange(img, "(h2 h) w -> h2 h w", h2=2)
print("result shape:", y.shape)  # (2, 2, 6)
y
```

## 2. `reduce` — aggregate over axes

`reduce` collapses axes with an aggregation (`sum`, `mean`, `max`, ...).
Axes present on the left but absent on the right are reduced over.

```{code-cell} ipython3
x = jnp.arange(24).reshape(2, 3, 4)
x
```

### 2a. Sum over one axis: `'a b c -> a c', 'sum'`

Summing out axis $b$:

$$
Y_{a,c} = \sum_{b=0}^{B-1} X_{a,b,c}.
$$

```{code-cell} ipython3
y = reduce(x, "a b c -> a c", "sum")
print("shape:", y.shape)  # (2, 4)
y
```

### 2b. Mean over two axes: `'a b c -> a', 'mean'`

Averaging over both $b$ and $c$:

$$
Y_{a} = \frac{1}{BC}\sum_{b=0}^{B-1}\sum_{c=0}^{C-1} X_{a,b,c}.
$$

```{code-cell} ipython3
# turn them into arrays of floats
a = a.astype('float32')
b = b.astype('float32')
x = x.astype('float32')
```

```{code-cell} ipython3
y = reduce(x, "a b c -> a", "mean")
print("shape:", y.shape)  # (2,)
y
```

### 2c. 2x2 max-pool: `'b c (h h2) (w w2) -> b c h w', 'max'`

A classic pooling pattern: split spatial axes into blocks, then take the
max within each block. With block size $2\times 2$,

$$
Y_{b,c,h,w} = \max_{\,0 \le i,j < 2}\; X_{b,\,c,\,2h+i,\,2w+j}.
$$

On the $4\times 4$ input below, each $2\times 2$ block keeps its largest
entry (its bottom-right corner):

$$
\left[\begin{array}{cc|cc}
0 & 1 & 2 & 3 \\
4 & \mathbf{5} & 6 & \mathbf{7} \\ \hline
8 & 9 & 10 & 11 \\
12 & \mathbf{13} & 14 & \mathbf{15}
\end{array}\right]
\;\longmapsto\;
\begin{bmatrix} 5 & 7 \\ 13 & 15 \end{bmatrix}.
$$

```{code-cell} ipython3
feat = jnp.arange(1 * 1 * 4 * 4).reshape(1, 1, 4, 4)
print("input:\n", feat[0, 0])
y = reduce(feat, "b c (h h2) (w w2) -> b c h w", "max", h2=2, w2=2)
print("pooled shape:", y.shape)  # (1, 1, 2, 2)
y[0, 0]
```

## 3. `repeat` — copy data along axes

`repeat` introduces new axes (or enlarges existing ones) by copying.

+++

### 3a. New leading axis: `'c -> h c', h=2`

Broadcasting a vector into $H$ identical rows:

$$
Y_{h,c} = v_{c} \quad \text{for all } 0 \le h < H,
\qquad
\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}
\;\longmapsto\;
\begin{bmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \end{bmatrix}.
$$

```{code-cell} ipython3
v = jnp.array([1, 2, 3])
y = repeat(v, "c -> h c", h=2)
print("shape:", y.shape)  # (2, 3)
y
```

## 4. Multi-head attention with `rearrange` + `einsum`

Putting it together. Let $x \in \mathbb{R}^{B\times S\times D}$ with
$H$ heads and head dimension $d = D/H$.

**Split heads.** Reshape $D = H\cdot d$ and move the head axis forward
(a split, as in {eq}`eq-merge` run backwards):

$$
Q_{b,h,i,d'} = x^{(q)}_{b,\,i,\,hd + d'}\quad\text{(and likewise }K,V\text{).}
$$

**Scores.** Scaled dot product, contracting the head dimension $d$:

```{math}
:label: eq-scores
S_{b,h,i,j} = \frac{1}{\sqrt{d}} \sum_{d'=0}^{d-1} Q_{b,h,i,d'}\,K_{b,h,j,d'}.
```

**Softmax** over the key axis $j$:

$$
A_{b,h,i,j} = \frac{\exp(S_{b,h,i,j})}{\sum_{j'} \exp(S_{b,h,i,j'})}.
$$

**Weighted values**, contracting the key axis $j$:

```{math}
:label: eq-values
O_{b,h,i,d'} = \sum_{j} A_{b,h,i,j}\, V_{b,h,j,d'}.
```

**Merge heads** back to $D$:

$$
\text{out}_{b,i,\,hd + d'} = O_{b,h,i,d'}.
$$

For a single batch and head, {eq}`eq-scores` and {eq}`eq-values` are two
ordinary matrix products, which is why each is one `einsum`:

$$
S = \frac{1}{\sqrt{d}}\, Q K^{\mathsf T} \quad (S\times S),
\qquad
O = A\, V \quad (S\times d).
$$

```{code-cell} ipython3
batch, seq, dim = 1, 4, 8
heads = 2
head_dim = dim // heads

x = jnp.ones((batch, seq, dim))
q = k = v = x  # pretend these came from linear projections

# split heads: (b, s, (h d)) -> (b, h, s, d)
q = rearrange(q, "b s (h d) -> b h s d", h=heads)
k = rearrange(k, "b s (h d) -> b h s d", h=heads)
v = rearrange(v, "b s (h d) -> b h s d", h=heads)
print("q heads shape:", q.shape)  # (1, 2, 4, 4)

# scores: contract over d
scores = jnp.einsum("b h i d, b h j d -> b h i j", q, k) / (head_dim**0.5)
print("scores shape:", scores.shape)  # (1, 2, 4, 4)

attn = nn.softmax(scores, axis=-1)

# weighted sum of values: contract over j
out = jnp.einsum("b h i j, b h j d -> b h i d", attn, v)
print("out (per head) shape:", out.shape)  # (1, 2, 4, 4)

# merge heads: (b, h, s, d) -> (b, s, (h d))
out = rearrange(out, "b h s d -> b s (h d)")
print("out merged shape:", out.shape)  # (1, 4, 8)
out
```

### Note on the `einsum` strings

Each `einsum` subscript mirrors its summation exactly:

- `'b h i d, b h j d -> b h i j'` — the index $d$ appears on both inputs
  but not the output, so it is summed over; $i,j$ are kept. This is {eq}`eq-scores`.
- `'b h i j, b h j d -> b h i d'` — now $j$ is the repeated-and-dropped
  index, so it is the one contracted. This is {eq}`eq-values`.

```{tip}
A repeated index that is absent from the output is summed; everything
else is carried through. That single rule is the whole of `einsum`.
```

+++

Examples from [Einsum is All you Need](https://rockt.ai/2018/04/30/einsum) case studies.

TreeQN (study 3.1)
For layer $l$, let $z_l$ be a low-dimensional state representation with transition function $W^a$ per action $a$. We caculate all next-state representations $z^{a}_{l+1} using a residual connection.

$$
    z^{a}_{l+1} = z_l + \tanh(W^a z_l)
$$
We do this for batch $B$ of $K$-dimensional $z \in \mathbb{R}^{B \times K}$ and for all transition functions at the same time. these functions are all arranged in a tensor $\hat{W} \in \mathbb{R}^{A \times K \times K}$ and calculate them as below.
```python
import torch.nn.functional as F

def random_tensors(shape, num=1, requires_grad=False):
  tensors = [torch.randn(shape, requires_grad=requires_grad) for i in range(0, num)]
  return tensors[0] if num == 1 else tensors

# Parameters
# -- [num_actions x hidden_dimension]
b = random_tensors([5, 3], requires_grad=True)
# -- [num_actions x hidden_dimension x hidden_dimension]
W = random_tensors([5, 3, 3], requires_grad=True)

def transition(zl):
  # -- [batch_size x num_actions x hidden_dimension]
  return zl.unsqueeze(1) + F.tanh(torch.einsum("bk,aki->bai", [zl, W]) + b)

# Sampled dummy inputs
# -- [batch_size x hidden_dimension]
zl = random_tensors([2, 3])

transition(zl)
```

```{code-cell} ipython3
try:
  import torch
  import torch.nn.functional as F
except ImportError:
  print("torch not installed")
  raise

def random_tensors(shape, num=1, requires_grad=False):
  tensors = [torch.randn(shape, requires_grad=requires_grad) for i in range(0, num)]
  return tensors[0] if num == 1 else tensors

# Parameters
# -- [num_actions x hidden_dimension]
b = random_tensors([5, 3], requires_grad=True)
# -- [num_actions x hidden_dimension x hidden_dimension]
W = random_tensors([5, 3, 3], requires_grad=True)

def transition(zl):
  # -- [batch_size x num_actions x hidden_dimension]
  return zl.unsqueeze(1) + F.tanh(torch.einsum("bk,aki->bai", [zl, W]) + b)

# Sampled dummy inputs
# -- [batch_size x hidden_dimension]
zl = random_tensors([2, 3])

transition(zl)
```

```{code-cell} ipython3

```
