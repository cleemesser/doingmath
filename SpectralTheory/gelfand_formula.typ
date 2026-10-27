// Gelfand's spectral radius formula, illustrated with worked examples.
//
// A Calepin computational notebook: the ```python blocks below are executed by
// the `calepin` CLI (one persistent Python session for the whole file) and their
// printed output and matplotlib figures are injected into the document.
// Plain `typst compile` shows the code without running it.
//
//   uv run calepin compile SpectralTheory/gelfand_formula.typ          # -> PDF
//   uv run calepin compile SpectralTheory/gelfand_formula.typ --format html  # -> HTML
//
// `uv run` matters: Calepin starts `python3` from PATH (or $CALEPIN_PYTHON), and
// that has to be the project venv so numpy/scipy/matplotlib import.

#import "/.calepin/calepin.typ" as calepin
#show: calepin.document

#set document(title: [Gelfand's Formula: Where the Spectral Radius Comes From])
#set math.equation(numbering: "(1)")

#calepin.setup(
  echo: true,
  eval: true,
  results: "verbatim",
  results-location: "chunk",
  fenced-chunks: true,
  fig-device-width: 9,
  fig-device-height: 3.6,
)

#let py = calepin.inline.with("python")

#title()

The spectral radius of a square matrix $A$ is the largest eigenvalue modulus,
$rho(A) = max_(lambda in sigma(A)) abs(lambda)$. Gelfand's formula says that this purely
algebraic quantity can be read off from the _growth of the powers_ of $A$, measured in any norm:

$ rho(A) = lim_(k -> oo) norm(A^k)^(1\/k). $ <eq-gelfand>

For a submultiplicative norm ($norm(A B) <= norm(A) norm(B)$, which every induced operator norm
satisfies) there is also a bound at every finite $k$:

$ rho(A) <= norm(A^k)^(1\/k) quad "for all" k >= 1, $ <eq-bound>

because $rho(A)^k = rho(A^k) <= norm(A^k)$: if $A^k v = lambda^k v$ then
$abs(lambda)^k norm(v) = norm(A^k v) <= norm(A^k) norm(v)$.

The notebook works through examples in order of difficulty. Each one isolates one reason why
$norm(A^k)^(1\/k)$ can differ from $rho(A)$ at finite $k$, and shows the $k$-th root removing it.

+ A scalar, then normal matrices: no gap at all.
+ A Jordan block: a _polynomial_ factor $k^(m-1)$.
+ A non-normal but diagonalizable matrix: a _constant_ factor (the eigenvector conditioning).
+ Different norms, including one that is not submultiplicative.
+ Subadditivity (Fekete's lemma): why the limit exists, and why it is an infimum.
+ Gelfand's own proof: the resolvent and its radius of convergence.
+ Practice: estimating $rho$ by repeated squaring, and its failure modes.
+ The dual view: a norm adapted to $A$ (a Lyapunov norm) in which $norm(A)$ is close to $rho(A)$.

= Setup

The helpers compute $norm(A^k)^(1\/k)$ by multiplying by $A$ one step at a time and
renormalizing after each step, so that $A^k$ never overflows or underflows. Write
$M_k = A^k \/ S_k$ with $S_k = norm(A^k)$. Then $M_k = (M_(k-1) A) \/ s_k$ with
$s_k = norm(M_(k-1) A)$, so $log S_k = sum_(i <= k) log s_i$, and the estimate is
$exp(log S_k \/ k)$.

```python
#| echo: true
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov

np.set_printoptions(precision=4, suppress=True)

plt.rcParams.update(
    {
        "figure.facecolor": "#0f0f0f",
        "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#ccc",
        "text.color": "#eee",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "monospace",
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444",
    }
)

BLUE = "#4fc3f7"
ORANGE = "#ffb74d"
GREEN = "#81c784"
RED = "#ef5350"
PURPLE = "#ce93d8"
GRAY = "#9e9e9e"


def op2(M):
    """Spectral norm (largest singular value), the default norm in this notebook."""
    return np.linalg.norm(M, 2)


def rho(A):
    """Spectral radius from the eigenvalues: the reference answer."""
    return np.max(np.abs(np.linalg.eigvals(A)))


def gelfand_sequence(A, kmax, norm=op2):
    """Return ||A^k||^(1/k) for k = 1..kmax, renormalizing A^k at every step."""
    M = np.eye(A.shape[0], dtype=A.dtype)
    log_S = 0.0
    out = np.empty(kmax)
    for k in range(1, kmax + 1):
        M = M @ A
        s = norm(M)
        log_S += np.log(s)
        M = M / s
        out[k - 1] = np.exp(log_S / k)
    return out


ks = np.arange(1, 201)
```

= Example 1: a scalar, and normal matrices

For a $1 times 1$ matrix $A = (a)$, $norm(A^k)^(1\/k) = abs(a^k)^(1\/k) = abs(a)$ at every $k$. There is
nothing to converge.

The same is true for any _normal_ matrix ($A A^* = A^* A$) in the spectral norm. A normal matrix is
unitarily diagonalizable, $A = U Lambda U^*$, and the spectral norm is unitarily invariant, so
$norm(A^k)_2 = norm(Lambda^k)_2 = rho(A)^k$ exactly. Two examples are a diagonal matrix and a
rotation scaled by $r$, whose eigenvalues are $r e^(plus.minus i theta)$:

$ D = mat(0.9, 0; 0, -0.5), quad R = r mat(cos theta, -sin theta; sin theta, cos theta). $

```python
a = -0.8
scalar_seq = gelfand_sequence(np.array([[a]]), 50)
print("scalar: ||a^k||^(1/k) == |a| for all k:", np.allclose(scalar_seq, abs(a)))

D = np.diag([0.9, -0.5])
r, theta = 0.9, 0.7
R = r * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

for name, A in [("D", D), ("R", R)]:
    seq = gelfand_sequence(A, 200)
    normal = np.allclose(A @ A.T, A.T @ A)
    print(f"{name}: normal? {normal}  rho = {rho(A):.4f}  ||A||_2 = {op2(A):.4f}")
    print(f"   ||A^k||^(1/k) == rho for k = 1..200: {np.allclose(seq, rho(A))}")
```

So for normal matrices @eq-bound is an equality at every $k$, and $norm(A)_2$ is already the
spectral radius. Every gap in the examples below comes from non-normality.

= Example 2: a Jordan block, where the gap is a polynomial in $k$

The simplest non-normal matrix is a $2 times 2$ Jordan block. Its powers have a closed form,
because $J = lambda I + N$ with $N^2 = 0$ and $lambda I$ commuting with $N$:

$ J = mat(lambda, 1; 0, lambda), quad
  J^k = (lambda I + N)^k = lambda^k I + k lambda^(k-1) N = mat(lambda^k, k lambda^(k-1); 0, lambda^k). $

Both eigenvalues equal $lambda$, so $rho(J) = abs(lambda)$. The off-diagonal entry
$k lambda^(k-1)$ grows before it decays. With $lambda = 0.9$ it peaks near $k = -1 \/ ln lambda approx 9.5$,
where $norm(J^k)_2 approx 3.9$. So $norm(J^k)$ shows a _transient hump_ even though every eigenvalue is inside the
unit disc.

For a block of size $m$ the same binomial expansion stops at $N^(m-1)$:

$ (lambda I + N)^k = sum_(j=0)^(m-1) binom(k, j) lambda^(k-j) N^j, $

so the largest entry grows like $k^(m-1) abs(lambda)^k$, and

$ norm(J^k)^(1\/k) approx abs(lambda) dot.c (C k^(m-1))^(1\/k) -> abs(lambda), $

since $C^(1\/k) -> 1$ and $k^((m-1)\/k) = e^((m-1) ln k \/ k) -> 1$. The relative error decays like
$(m-1) ln k \/ k$, which is _slow_.

```python
lam = 0.9
J = np.array([[lam, 1.0], [0.0, lam]])

# The closed form for J^k against repeated multiplication.
Jk_closed = lambda k: np.array([[lam**k, k * lam ** (k - 1)], [0.0, lam**k]])
print(
    "J^k closed form matches matrix_power for k = 1..60:",
    all(np.allclose(Jk_closed(k), np.linalg.matrix_power(J, k)) for k in range(1, 61)),
)


def jordan_block(lam, m):
    return lam * np.eye(m) + np.eye(m, k=1)


fig, (ax0, ax1) = plt.subplots(1, 2)
norms = [op2(np.linalg.matrix_power(J, k)) for k in ks[:60]]
ax0.plot(ks[:60], norms, color=BLUE, label=r"$\|J^k\|_2$")
ax0.plot(ks[:60], lam ** ks[:60], color=ORANGE, ls="--", label=r"$\rho(J)^k$")
ax0.set_xlabel("k")
ax0.set_title("transient growth, then decay")
ax0.legend()
ax0.grid(True)

for m, color in zip([1, 2, 3, 4], [GREEN, BLUE, PURPLE, RED]):
    seq = gelfand_sequence(jordan_block(lam, m), len(ks))
    ax1.semilogy(ks, seq / lam - 1 + 1e-17, color=color, label=f"m = {m}")
    if m > 1:
        ax1.semilogy(ks, (m - 1) * np.log(ks) / ks, color=color, ls=":", lw=1)
ax1.set_xlabel("k")
ax1.set_title(r"relative error $\|J^k\|^{1/k}/\rho - 1$")
ax1.set_ybound(1e-3, 3)
ax1.legend(loc="upper right")
ax1.grid(True)
fig.tight_layout()
```

The dotted curves in the right panel are $(m-1) ln k \/ k$. They track the slope of each solid curve:
on these axes the error falls by roughly one decade per decade of $k$, minus a logarithm. The $m=1$
block is the scalar $lambda$ and has zero error, so it does not appear on the log axis.

= Example 3: a non-normal, diagonalizable matrix, where the gap is a constant

Now take distinct eigenvalues but a large off-diagonal coupling:

$ A_c = mat(0.9, c; 0, 0.5), quad rho(A_c) = 0.9 "for every" c. $

$A_c$ is diagonalizable, $A_c = V Lambda V^(-1)$, so

$ norm(A_c^k) = norm(V Lambda^k V^(-1)) <= kappa(V) rho^k,
  quad "hence" quad rho <= norm(A_c^k)^(1\/k) <= kappa(V)^(1\/k) rho, $

where $kappa(V) = norm(V) norm(V^(-1))$ is the condition number of the eigenvector matrix. As $c$
grows the two eigenvectors become nearly parallel and $kappa(V)$ blows up. The spectral radius
does not move, but $norm(A_c)_2$, the $k=1$ estimate, grows like $c$. The constant $kappa(V)$ costs
a relative error of about $ln kappa(V) \/ k$, which is still slow, but there is no extra $ln k$.

```python
cs = [0.0, 1.0, 10.0, 50.0]
colors = [GREEN, BLUE, PURPLE, RED]

fig, (ax0, ax1) = plt.subplots(1, 2)
for c, color in zip(cs, colors):
    A = np.array([[0.9, c], [0.0, 0.5]])
    evals, V = np.linalg.eig(A)
    kappa = np.linalg.cond(V)
    seq = gelfand_sequence(A, len(ks))
    assert np.all(seq >= rho(A) - 1e-12)  # lower bound, eq. (2)
    assert np.all(seq <= kappa ** (1 / ks) * rho(A) + 1e-12)  # kappa(V) upper bound
    print(
        f"c = {c:4.0f}:  ||A||_2 = {op2(A):6.2f}   kappa(V) = {kappa:6.1f}"
        f"   ||A^128||^(1/128) = {seq[127]:.4f}"
    )
    ax0.plot(ks, seq, color=color, label=f"c = {c:g}")
    ax1.loglog(ks, seq / 0.9 - 1 + 1e-17, color=color)
    if c > 0:
        ax1.loglog(ks, np.log(kappa) / ks, color=color, ls=":", lw=1)
ax0.axhline(0.9, color=ORANGE, ls="--", lw=1, label=r"$\rho = 0.9$")
ax0.set_ybound(0.8, 2.0)
ax0.set_xlabel("k")
ax0.set_title(r"$\|A_c^k\|_2^{1/k}$")
ax0.legend()
ax0.grid(True)
ax1.set_ybound(1e-4, 10)
ax1.set_xlabel("k")
ax1.set_title(r"relative error vs. $\ln\kappa(V)/k$ (dotted)")
ax1.grid(True)
fig.tight_layout()
```

Both bounds hold at every $k$ (the `assert`s pass), and for $c = 50$ the estimate falls from about
$50$ at $k = 1$ to within a few percent of $rho$ by $k = 128$. Example 2 and Example 3 between them
account for the asymptotic formula $norm(A^k) approx C k^(m-1) rho^k$ in general: $m$ is the size of
the largest Jordan block among the eigenvalues of modulus $rho$, and $C$ absorbs the conditioning
of the Jordan basis.

= Example 4: every norm gives the same limit

@eq-gelfand holds for _any_ norm on the space of $n times n$ matrices. In finite dimensions all
norms are equivalent: for any two there are constants with
$a norm(X)_alpha <= norm(X)_beta <= b norm(X)_alpha$. Taking $k$-th roots sends $a^(1\/k)$ and
$b^(1\/k)$ to 1, so all norms share one limit. The finite-$k$ bound @eq-bound, on the other hand,
needs submultiplicativity.

The largest-entry norm $norm(X)_max = max_(i j) abs(x_(i j))$ is a genuine norm that is not
submultiplicative. On the matrix of all one-halves,

$ B = 1/2 mat(1, 1; 1, 1), quad B^2 = B, quad rho(B) = 1, quad norm(B^k)_max = 1/2 < rho(B), $

so @eq-bound fails at every $k$. Still, $(1\/2)^(1\/k) -> 1 = rho(B)$, as @eq-gelfand promises.

```python
norms = {
    "spectral": op2,
    "max column sum": lambda M: np.linalg.norm(M, 1),
    "max row sum": lambda M: np.linalg.norm(M, np.inf),
    "Frobenius": lambda M: np.linalg.norm(M, "fro"),
    "max entry*": lambda M: np.max(np.abs(M)),  # * not submultiplicative
}

A = np.array([[0.9, 10.0], [0.0, 0.5]])
B = np.full((2, 2), 0.5)
print(f"B @ B == B: {np.allclose(B @ B, B)}   rho(B) = {rho(B):.4f}")

fig, (ax0, ax1) = plt.subplots(1, 2)
for (name, nrm), color in zip(norms.items(), [BLUE, GREEN, PURPLE, ORANGE, RED]):
    seqA = gelfand_sequence(A, len(ks), nrm)
    seqB = gelfand_sequence(B, len(ks), nrm)
    print(
        f"{name:15s} A_10 at k=200: {seqA[-1]:.4f}"
        f"    B: min_k {seqB.min():.4f}, at k=200 {seqB[-1]:.4f}"
    )
    ax0.plot(ks, seqA, color=color, label=name)
    ax1.plot(ks, seqB, color=color, label=name)
ax0.axhline(rho(A), color="w", ls="--", lw=1)
ax0.set_ybound(0.6, 2.0)
ax0.set_title(r"$A_{10}$: every norm $\to \rho = 0.9$")
ax0.set_xlabel("k")
ax0.legend(fontsize=7)
ax0.grid(True)
ax1.axhline(rho(B), color="w", ls="--", lw=1)
ax1.set_title(r"$B$: max-entry norm sits below $\rho$")
ax1.set_xlabel("k")
ax1.grid(True)
fig.tight_layout()
```

For $A_(10)$ the five curves differ only at small $k$; the norm-equivalence constants are
washed out by the $k$-th root. For $B$ the four submultiplicative norms give exactly $1$ at every
$k$, while the max-entry norm starts at $1\/2$, below $rho$, and climbs as $2^(-1\/k)$.

= Example 5: why the limit exists — subadditivity and Fekete's lemma

For a submultiplicative norm put $a_k = log norm(A^k)$. Then

$ a_(j+k) = log norm(A^j A^k) <= log norm(A^j) + log norm(A^k) = a_j + a_k, $

so the sequence $a_k$ is _subadditive_. Fekete's lemma (1923) says that for any subadditive sequence
$a_k \/ k$ converges, and its limit is $inf_k a_k \/ k$. So the limit in @eq-gelfand exists _and_
is the infimum over all $k$, without any reference to eigenvalues:

$ lim_(k -> oo) norm(A^k)^(1\/k) = inf_(k >= 1) norm(A^k)^(1\/k). $

Fekete's argument proves that the limit exists, but not that the limit equals $rho(A)$; that
identification is Gelfand's theorem (Example 6). It also does not make the sequence decreasing.
The matrix

$ S = mat(0, 4; 1\/4, 0), quad S^2 = I, quad rho(S) = 1 $

gives $norm(S^k)_2 = 1$ for even $k$ and $4$ for odd $k$, so the sequence is the sawtooth
$4^(1\/k), 1, 4^(1\/3), 1, dots$. Along the powers of two, however, it _is_ non-increasing, because
$norm(A^(2k)) <= norm(A^k)^2$ gives $norm(A^(2k))^(1\/2k) <= norm(A^k)^(1\/k)$. That is why
estimating by repeated squaring ($k = 1, 2, 4, dots, 2^j$) can only improve at each step.

```python
S = np.array([[0.0, 4.0], [0.25, 0.0]])
seqS = gelfand_sequence(S, 20)
print("S @ S == I:", np.allclose(S @ S, np.eye(2)), "  rho(S) =", rho(S))
print("||S^k||^(1/k), k = 1..8:", seqS[:8])

# Monotone along k = 2^j, for many random non-normal matrices.
rng = np.random.default_rng(0)
ok = True
for _ in range(200):
    n = rng.integers(2, 8)
    V = rng.normal(size=(n, n))
    A = V @ np.diag(rng.uniform(0.2, 1.5, n)) @ np.linalg.inv(V)
    seq = gelfand_sequence(A, 256)
    pow2 = seq[[2**j - 1 for j in range(9)]]  # k = 1, 2, 4, ..., 256
    ok &= bool(np.all(np.diff(pow2) <= 1e-9 * pow2[:-1]))  # monotone along 2^j
    ok &= bool(seq.min() >= rho(A) * (1 - 1e-9))  # inf_k still bounded below by rho
print("non-increasing along k = 2^j, and >= rho, for 200 random matrices:", ok)

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(np.arange(1, 21), seqS, "o-", color=BLUE, ms=4, label=r"$\|S^k\|_2^{1/k}$")
ax.axhline(1.0, color=ORANGE, ls="--", lw=1, label=r"$\rho(S) = 1$")
ax.set_xticks(range(1, 21))
ax.set_xlabel("k")
ax.legend()
ax.grid(True)
fig.tight_layout()
```

= Example 6: Gelfand's proof — the resolvent and its radius of convergence

Gelfand proved @eq-gelfand in a general Banach algebra (bounded operators, convolution algebras),
where there is no Jordan form. His argument is complex analysis.

*The resolvent as a power series.* For $abs(z)$ large the Neumann series gives

$ R(z) = (z I - A)^(-1) = sum_(k=0)^oo A^k / z^(k+1), $ <eq-neumann>

a Laurent series in $1\/z$ whose coefficients are matrices.

*Its radius of convergence.* By the Cauchy–Hadamard formula, @eq-neumann converges for
$abs(z) > r$ and diverges for $abs(z) < r$, where $r = limsup_(k -> oo) norm(A^k)^(1\/k)$. This is
exactly the right side of @eq-gelfand.

*Analyticity.* $R(z)$ is analytic everywhere off the spectrum $sigma(A)$ and blows up on it, since
$norm(R(z)) >= 1 \/ "dist"(z, sigma(A))$. A Laurent series about $oo$ converges on the largest
punctured disc around $oo$ in which the function is analytic, which is $abs(z) > rho(A)$. Hence $r = rho(A)$.

So the spectral radius _is_ a radius of convergence. Eigenvalues are the singularities of $R(z)$,
and the growth of $norm(A^k)$ is the growth of its Laurent coefficients.

The left panel below shows $log_(10) norm(R(z))_2$ over the complex plane for $A_(10)$. The
eigenvalues $0.5$ and $0.9$ are the poles. The dashed circle is $abs(z) = rho$, outside which
@eq-neumann converges. Convergence is obvious only for $abs(z) > norm(A)_2 approx 10$, far off the
plot, where $norm(A^k) <= norm(A)^k$ bounds the terms by a geometric series. In the whole region
between $rho$ and $norm(A)_2$ the series converges too, though the triangle inequality alone
cannot show it. The level sets of $norm(R(z))$ are the
_pseudospectra_ of $A$. They spread far beyond the eigenvalues, and that spread is the same
non-normality that causes the transient growth of $norm(A^k)$.

The right panel shows the size of the terms $norm(A^k) \/ abs(z)^(k+1)$ for real $z$ on either side
of $rho$.

```python
#| fig-device-height: 4.2
A = np.array([[0.9, 10.0], [0.0, 0.5]])
I2 = np.eye(2)


def resolvent_partial_sum(A, z, K):
    total, term = np.zeros_like(A, dtype=complex), np.eye(A.shape[0], dtype=complex) / z
    for _ in range(K):
        total += term
        term = term @ A / z
    return total


z_out = 0.95  # rho < |z| < ||A||_2: converges, though not obviously
exact = np.linalg.inv(z_out * I2 - A)
converged = np.allclose(resolvent_partial_sum(A, z_out, 2000), exact)
print("z = 0.95: partial sums -> (zI - A)^-1:", converged)
for K in (50, 100, 200):
    size = op2(resolvent_partial_sum(A, 0.85, K))
    print(f"z = 0.85: ||partial sum, K = {K:3d}||_2 = {size:.3g}")

x = np.linspace(-1.6, 1.6, 320)  # even count: no grid point lands on an eigenvalue
Z = x[None, :] + 1j * x[:, None]
M = Z[..., None, None] * I2 - A  # stack of (zI - A)
smin = np.linalg.svd(M, compute_uv=False)[..., -1]
logR = np.log10(1 / smin)  # ||(zI - A)^-1||_2 = 1 / sigma_min(zI - A)

fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
extent = [x[0], x[-1], x[0], x[-1]]
im = ax0.imshow(logR, extent=extent, origin="lower", cmap="magma", vmin=0, vmax=3)
ax0.contour(x, x, logR, levels=[1, 2], colors=[GRAY], linewidths=0.6)
t = np.linspace(0, 2 * np.pi, 400)
ax0.plot(rho(A) * np.cos(t), rho(A) * np.sin(t), color=BLUE, ls="--", label=r"$|z| = \rho$")
ev = np.linalg.eigvals(A)
ax0.plot(ev.real, ev.imag, "x", color=GREEN, ms=8, mew=2, label="eigenvalues")
ax0.set_title(r"$\log_{10}\|(zI-A)^{-1}\|_2$  ($\|A\|_2 \approx 10$ is off-plot)", fontsize=9)
ax0.set_xlabel("Re z")
ax0.set_ylabel("Im z")
ax0.legend(fontsize=7, loc="lower left")
fig.colorbar(im, ax=ax0, fraction=0.046)

powers = np.array([op2(np.linalg.matrix_power(A, k)) for k in range(150)])
for z, color in [(0.85, RED), (0.9, ORANGE), (0.95, GREEN), (1.2, BLUE)]:
    ax1.semilogy(powers / z ** (np.arange(150) + 1), color=color, label=f"z = {z}")
ax1.set_xlabel("k")
ax1.set_title(r"terms $\|A^k\| / |z|^{k+1}$ of the Neumann series", fontsize=9)
ax1.legend(fontsize=7)
ax1.grid(True)
fig.tight_layout()
```

The terms decay geometrically for $abs(z) > rho$. At $abs(z) = rho$ they settle to a constant, and
for $abs(z) < rho$ they grow, whatever the transient does first. The same argument with Liouville's theorem also shows that the spectrum is
never empty: if it were, $R(z)$ would be an entire function tending to $0$ at $oo$, hence
identically zero, which is impossible.

= Example 7: estimating $rho$ in practice by repeated squaring

To estimate $rho$ without an eigensolver one squares $j$ times to get $A^(2^j)$, using
$j$ matrix products for $k = 2^j$. Two practical points follow from the examples above.

*Overflow and underflow.* Forming $A^(128)$ directly overflows in single precision once $rho gt.tilde 2$
($2^(128) approx 3.4 times 10^(38)$ is the `float32` limit), and underflows to $0$ when $rho$ is small.
The remedy is the same renormalization as in `gelfand_sequence`. Put $M_0 = A \/ norm(A)$ and
$M_(i+1) = M_i^2 \/ norm(M_i^2)$. If $r_i$ denotes the estimate at $k = 2^i$, then
$log r_(i+1) = log r_i + log norm(M_i^2) \/ 2^(i+1)$.

```python
def gelfand_by_squaring(A, j, norm=op2):
    """Estimates ||A^(2^i)||^(1/2^i) for i = 0..j by j renormalized squarings."""
    s = norm(A)
    M, log_r = A / s, np.log(s)
    out = [np.exp(log_r)]
    for i in range(j):
        M = M @ M
        s = norm(M)
        M, log_r = M / s, log_r + np.log(s) / 2 ** (i + 1)
        out.append(np.exp(log_r))
    return np.array(out)


A = np.array([[0.9, 10.0], [0.0, 0.5]])
at_powers_of_2 = gelfand_sequence(A, 128)[[2**i - 1 for i in range(8)]]
print("squaring estimates equal gelfand_sequence at k = 2^i:",
      np.allclose(gelfand_by_squaring(A, 7), at_powers_of_2))

for scale in (2.5, 0.3):
    A32 = (scale / 0.9 * A).astype(np.float32)  # same shape, rho rescaled to 2.5 and 0.3
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        A128 = np.linalg.matrix_power(A32, 128)  # computed in float32
    naive = op2(A128.astype(np.float64)) ** (1 / 128) if np.all(np.isfinite(A128)) else np.inf
    safe = gelfand_by_squaring(A32, 7)[-1]
    print(f"float32, rho = {rho(A32):.2f}:  naive = {naive:.4f}   renormalized = {safe:.4f}")
```

*A margin above $rho$.* Even at $k = 128$ the estimate exceeds $rho$ by the factor
$(C k^(m-1))^(1\/k)$, for instance $50^(1\/128) approx 1.031$. A stabilizing clip of the form
$A <- A \/ max(1, hat(rho))$, with $hat(rho) = norm(A^(128))^(1\/128)$, therefore also shrinks
_stable_ operators whose $rho$ lies within that margin of 1. The table scans $rho$ toward 1 for a
mildly and a strongly non-normal family.

```python
print(f"{'rho':>6} | {'c':>5} | {'||A||_2':>8} | {'est k=128':>9} | clip fires?")
for c in (1.0, 50.0):
    for r in (0.90, 0.95, 0.97, 0.98, 0.99):
        A = np.array([[r, c], [0.0, 0.5]])
        est = gelfand_by_squaring(A, 7)[-1]
        fires = "yes" if est > 1 else "no"
        print(f"{r:6.2f} | {c:5.0f} | {op2(A):8.2f} | {est:9.4f} | {fires}")
```

With $c = 50$, operators with $rho$ between roughly $0.97$ and $1$ are flagged, even though they are
stable. A clip on $norm(A)_2$ would flag every operator in the table. When $n$ is small enough to
call an eigensolver, `max(abs(eigvals(A)))` removes the margin entirely.

= Example 8: the dual view — a norm adapted to $A$

Fekete gives $rho(A) = inf_k norm(A^k)^(1\/k)$ for one fixed norm. A complementary statement, due
to Householder, fixes $k = 1$ and varies the norm: for every $epsilon > 0$ there is an induced norm
with $norm(A) <= rho(A) + epsilon$. In short, $rho(A)$ is the infimum of $norm(A)$ over all
operator norms.

A constructive version uses a Lyapunov equation. If $rho(A) < 1$, the discrete Lyapunov equation

$ A^T P A - P = -I $

has a unique positive-definite solution $P$. In the norm $norm(x)_P = sqrt(x^T P x)$,

$ norm(A x)_P^2 = x^T P x - x^T x <= (1 - 1 \/ lambda_"max"(P)) norm(x)_P^2, $

so $A$ is a strict contraction, $norm(A)_P <= sqrt(1 - 1\/lambda_"max"(P)) < 1$. Applying the same
construction to $A \/ (rho + epsilon)$ gives a norm in which $norm(A) <= rho + epsilon$.
The price is the conditioning of $P$: the non-normality constant $C$ from Example 3 has moved
into the geometry of the norm. With $P = L L^T$, the induced norm is
$norm(A)_P = norm(L^T A L^(-T))_2$.

```python
def lyapunov_norm(A):
    """Solve A^T P A - P = -I; return ||A||_P, cond(P), sqrt(1 - 1/lambda_max(P))."""
    P = solve_discrete_lyapunov(A.T, np.eye(A.shape[0]))  # solves A^T P A - P + I = 0
    assert np.allclose(A.T @ P @ A - P, -np.eye(A.shape[0]))
    L = np.linalg.cholesky(P)
    normP = op2(L.T @ A @ np.linalg.inv(L.T))
    return normP, np.linalg.cond(P), np.sqrt(1 - 1 / np.linalg.eigvalsh(P).max())


A = np.array([[0.9, 50.0], [0.0, 0.5]])
nP, condP, bound = lyapunov_norm(A)
print(f"A_50: rho = {rho(A):.3f}   ||A||_2 = {op2(A):.2f}   cond(P) = {condP:.3g}")
print(f"      1 - ||A||_P = {1 - nP:.3g} >= 1 - bound = {1 - bound:.3g}  (barely)")

J = np.array([[0.9, 1.0], [0.0, 0.9]])
for name, M in [("A_50", A), ("Jordan block, lambda = 0.9", J)]:
    print(f"\n{name}")
    print(f"{'eps':>8} | {'rho + eps':>9} | {'||M||_P(eps)':>12} | {'cond(P)':>9}")
    for eps in (0.1, 0.03, 0.01, 0.003, 0.001):
        nP, condP, _ = lyapunov_norm(M / (rho(M) + eps))
        assert nP * (rho(M) + eps) <= rho(M) + eps
        r_eps = rho(M) + eps
        print(f"{eps:8.3f} | {r_eps:9.4f} | {nP * r_eps:12.4f} | {condP:9.3g}")

# A diagonalizable matrix attains the infimum:
# in ||x||_V = ||V^-1 x||_2 one has ||A||_V = ||Lambda||_2 = rho.
evals, V = np.linalg.eig(A)
print(f"\nA_50, eigenvector norm: ||V^-1 A V||_2 = {op2(np.linalg.inv(V) @ A @ V):.4f} = rho,"
      f"  kappa(V) = {np.linalg.cond(V):.0f}")
```

With $Q = I$ and no rescaling, $A_(50)$ is only barely a contraction: $1 - norm(A)_P$ is of
order $1\/lambda_"max"(P)$, which is tiny because $P$ is so ill-conditioned. The $epsilon$-scans
show the Householder statement at work. For both matrices $norm(M)_(P(epsilon))$ approaches $rho$
while $kappa(P)$ grows, roughly like $1\/epsilon$ for $A_(50)$ and like $1\/epsilon^2$ for the
Jordan block. The difference matters. A diagonalizable matrix _attains_ the infimum: in the norm
$norm(x)_V = norm(V^(-1) x)_2$ one has $norm(A)_V = norm(Lambda)_2 = rho(A)$, at the price
$kappa(V)$ from Example 3.
No norm achieves $norm(J) = rho(J)$ for a Jordan block: that would give $norm(J^k) <= rho^k$,
contradicting the growth $norm(J^k) approx k rho^(k-1)$ seen in Example 2, since all norms are
equivalent. The two
routes are therefore complementary. Gelfand fixes the norm and takes powers, paying
$(C k^(m-1))^(1\/k)$. Lyapunov and Householder take one power in an adapted norm, paying $kappa(P)$.

= Summary

- $rho(A) <= norm(A^k)^(1\/k)$ at every $k$ for submultiplicative norms. For the limit
  $norm(A^k)^(1\/k) -> rho(A)$, any norm will do.
- For normal matrices in the spectral norm the gap is zero. Otherwise the gap is
  $(C k^(m-1))^(1\/k)$: a constant from eigenvector conditioning (Example 3) and a polynomial from
  Jordan blocks (Example 2). The relative error decays only like $(ln C + (m-1) ln k)\/k$.
- Subadditivity of $log norm(A^k)$ makes the limit exist and equal the infimum over $k$. Along
  $k = 2^j$ the estimates never increase.
- Gelfand's insight: $rho(A)$ is the radius of convergence of the resolvent's Laurent series at
  $oo$, so a metric quantity (coefficient growth) equals an algebraic one (location of the
  singularities).
- In practice: renormalize while squaring, and remember the margin above $rho$ that remains at
  finite $k$.

= References

- I. Gelfand, "Normierte Ringe," _Rec. Math. [Mat. Sbornik] N.S._ 9(51):3–24, 1941. English
  translation in G. W. Mackey, _Commutative Banach Algebras_, Notas de Matemática 17, 1959.
- A. Beurling, "Sur les intégrales de Fourier absolument convergentes et leur application à une
  transformation fonctionnelle," _IX Congrès des Mathématiciens Scandinaves_, Helsingfors, 1938.
- M. Fekete, "Über die Verteilung der Wurzeln bei gewissen algebraischen Gleichungen mit
  ganzzahligen Koeffizienten," _Math. Z._ 17:228–249, 1923.
- J. Hadamard, "Essai sur l'étude des fonctions données par leur développement de Taylor,"
  _J. Math. Pures Appl._ (4) 8:101–186, 1892.
- A. S. Householder, _The Theory of Matrices in Numerical Analysis_, Blaisdell, 1964.
- R. A. Horn and C. R. Johnson, _Matrix Analysis_, 2nd ed., Cambridge University Press, 2013, §5.6.
- W. Rudin, _Functional Analysis_, 2nd ed., McGraw-Hill, 1991, Ch. 10.
- L. N. Trefethen and M. Embree, _Spectra and Pseudospectra: The Behavior of Nonnormal Matrices
  and Operators_, Princeton University Press, 2005.
