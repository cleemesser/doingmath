clm/Claude discussion:

Q: is there a convention to deal the sign ambiguity of eigenvects when doing
pca?

There's no single universal convention, but a few standard ones, roughly in order of how often you'll meet them:

**1. Max-abs element positive.** Flip each component so its largest-magnitude loading is positive. This is what scikit-learn does (`sklearn.utils.extmath.svd_flip`, applied to the left singular vectors by default). Deterministic and cheap, but unstable when two loadings are nearly tied in magnitude — a tiny perturbation flips the sign.

**2. First nonzero element positive.** Common in older LAPACK-adjacent code. Same fragility, worse.

**3. Sum of loadings positive** (equivalently, positive inner product with **1**). Fails badly for contrast-like components whose loadings sum to ~0.

**4. Data-driven: point the component the way the data points.** For component $k$ with scores $y_j = x_j^\top u_k$, set the sign to $\mathrm{sign}\!\left(\sum_j \mathrm{sign}(y_j)\, y_j^2\right)$. The squared weighting lets the high-leverage samples decide. This is the Bro–Acar–Kolda convention and it's the one I'd recommend as a default — it's stable, has an interpretable meaning ("the component points toward the bulk of the variance it explains"), and extends to Tucker/PARAFAC.

> Bro, Acar & Kolda (2008), *Resolving the sign ambiguity in the determination of factor models for two-way and three-way data*, J. Chemometrics 22:135–140. Also Sandia tech report SAND2007-6422.

**5. Reference alignment.** For bootstrap, cross-validation, or cross-subject comparisons, don't use an intrinsic rule — align to a reference solution by sign-flipping to maximize $\langle u_k, u_k^{\text{ref}}\rangle$, or do a full orthogonal Procrustes if you also want to handle rotation.

Two things worth flagging given your latent-space work. First, sign is only the
visible part of a gauge freedom: whenever eigenvalues are degenerate or
near-degenerate, you have full $O(m)$ rotational ambiguity within the
eigenspace, and no sign convention rescues you. There the honest move is to stop
comparing eigenvectors and compare subspaces — principal angles / Grassmannian
distance. Second, if eigenvectors are feeding a downstream model, the cleanest
fix is architectural invariance rather than a convention. See Lim et al., *Sign
and Basis Invariant Networks for Spectral Graph Representation Learning* (ICLR
2023, arXiv:2202.13013), which handles exactly the $u \mapsto -u$ and
eigenspace-basis ambiguities.

Q: Can you write a python function which implements 4. the Bro, Acar and Kolda algorithm

The checks pass: the output is invariant to whatever signs the solver happened to hand back, and the reconstruction $U\Sigma V^\top$ is unchanged.

Three things about the algorithm that are easy to miss from the paper:

**Both modes vote, and the pair is flipped together.** In a two-way model only the product $\mathrm{sign}(u_k)\cdot\mathrm{sign}(v_k)$ is fixed, so you can't orient scores and loadings independently. The algorithm computes evidence separately from each mode — $\sum_j \mathrm{sign}(a_j)a_j^2$ with $a = u_k^\top Y$ for the loadings side, and $a = Yv_k$ for the scores side — and when the two disagree, the one with larger magnitude overrules the other.

**Deflation is a no-op for PCA.** The paper deflates $Y_k = X - \sum_{j\neq k}\sigma_j u_j v_j^\top$ before scoring, but if $U$ has orthonormal columns then $u_k^\top Y_k = u_k^\top X$ exactly, since $u_k \perp u_j$. Deflation only earns its keep for oblique factor models — PARAFAC, Tucker with non-orthogonal factors, ICA. I kept the flag so the function generalizes, but for PCA you can set `deflate=False` and save the $O(npK)$ work.

**What the statistic reduces to.** Since $u_k^\top X = \sigma_k v_k^\top$ and $Xv_k = \sigma_k u_k$ exactly, the two votes collapse to $\sum_j \mathrm{sign}(v_{kj})v_{kj}^2$ on the loading vector and $\sum_i \mathrm{sign}(u_{ik})u_{ik}^2$ on the score vector — a skewness-like third-moment statistic on each. So the convention is really "orient the component so that its heavy tail points positive," on whichever side has the more decisive tail. That's also where it can be fragile: a component whose scores are near-symmetric and whose loadings are near-symmetric gives $s\approx 0$ on both sides, and the sign is then as arbitrary as it was before. Worth checking $|s_{\text{left}}|, |s_{\text{right}}|$ against the scale $\sigma_k^3$ if you care; if both are tiny, fall back to reference alignment against a template solution.

One caveat for your EEG/MRI pipelines: pass in the *same* preprocessed matrix used for the decomposition. If you centered and then scaled, score on the centered-and-scaled matrix, or the votes refer to a different geometry than the components do.

```python
"""
Bro, Acar & Kolda (2008) sign convention for PCA / truncated SVD.

Reference
---------
R. Bro, E. Acar, T. G. Kolda, "Resolving the sign ambiguity in the
determination of factor models for two-way and three-way data",
Journal of Chemometrics 22 (2008) 135-140.
"""

import numpy as np


def _signed_sq_sum(v):
    """sum_j sign(v_j) * v_j**2  -- squared weighting lets high-leverage
    entries dominate the vote."""
    return float(np.sum(np.sign(v) * v ** 2))


def resolve_signs_bak(X, U, S, Vt, deflate=True):
    """
    Fix the sign of each component of a factorization X ~= U diag(S) Vt.

    Parameters
    ----------
    X : (n, p) array
        The matrix that was factorized, preprocessed exactly as it was
        for the decomposition (i.e. already column-centered for PCA).
    U : (n, K), S : (K,), Vt : (K, p)
        Truncated SVD factors. S must be non-negative.
    deflate : bool
        Remove the other components' contribution before scoring each
        component. This is a no-op when U has orthonormal columns
        (as in PCA/SVD) and only matters for oblique factor models.

    Returns
    -------
    flips : (K,) array of +-1
    U_new, Vt_new : sign-corrected factors (S is unchanged)

    Notes
    -----
    Only the *pair* (u_k, v_k) has sign freedom: flipping one alone
    changes the model. So both modes vote, and if they disagree the
    weaker vote is overruled.
    """
    X = np.asarray(X, dtype=float)
    U = np.array(U, dtype=float, copy=True)
    S = np.asarray(S, dtype=float)
    Vt = np.array(Vt, dtype=float, copy=True)

    K = S.size
    flips = np.ones(K)
    recon = (U * S) @ Vt if deflate else None

    for k in range(K):
        if deflate:
            Y = X - recon + S[k] * np.outer(U[:, k], Vt[k])
        else:
            Y = X

        s_left = _signed_sq_sum(U[:, k] @ Y)   # project the p columns onto u_k
        s_right = _signed_sq_sum(Y @ Vt[k])    # project the n rows   onto v_k

        if s_left * s_right < 0:               # modes disagree: larger wins
            if abs(s_left) < abs(s_right):
                s_left = -s_left
            else:
                s_right = -s_right

        flips[k] = -1.0 if (s_left + s_right) < 0 else 1.0

    return flips, U * flips, Vt * flips[:, None]


def pca_bak(X, n_components=None, center=True):
    """PCA with BAK-canonical component signs.

    Returns (scores, loadings, explained_variance, mean).
    scores = U*S is (n, K); loadings = V is (p, K) with unit columns.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mean

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if n_components is not None:
        U, S, Vt = U[:, :n_components], S[:n_components], Vt[:n_components]

    _, U, Vt = resolve_signs_bak(Xc, U, S, Vt)
    var = S ** 2 / max(len(X) - 1, 1)
    return U * S, Vt.T, var, mean


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p, K = 200, 12, 4
    X = rng.normal(size=(n, K)) @ rng.normal(size=(K, p)) + 0.1 * rng.normal(size=(n, p))
    Xc = X - X.mean(0)

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U, S, Vt = U[:, :K], S[:K], Vt[:K]
    _, Uref, Vtref = resolve_signs_bak(Xc, U, S, Vt)

    # 1. invariance to the arbitrary signs the solver happened to return
    ok = True
    for _ in range(50):
        d = rng.choice([-1.0, 1.0], size=K)
        _, Ua, Vta = resolve_signs_bak(Xc, U * d, S, Vt * d[:, None])
        ok &= np.allclose(Ua, Uref) and np.allclose(Vta, Vtref)
    print("invariant to input sign flips:", ok)

    # 2. deflation really is a no-op for an orthogonal factorization
    _, Und, Vtnd = resolve_signs_bak(Xc, U, S, Vt, deflate=False)
    print("deflate flag irrelevant here:", np.allclose(Und, Uref))

    # 3. reconstruction is untouched
    print("model preserved:", np.allclose((Uref * S) @ Vtref, (U * S) @ Vt))

    # 4. agreement with sklearn's max-abs rule, for reference
    try:
        from sklearn.decomposition import PCA
        sk = PCA(n_components=K).fit(X)
        agree = np.sign(np.sum(sk.components_ * Vtref, axis=1))
        print("sign agreement with sklearn svd_flip:", agree)
    except ImportError:
        pass
```

Q: why is there an abs(s_left) < abs(s_right) in the signflip function in the paper I don't see that abs operation


You're right that I added it — I can't check the printed pseudocode from here, so let me give you the reasoning rather than claim the paper says something it may not.

Take the literal reading. You only reach that branch when `s1 * s2 < 0`, so exactly one of them is negative. In that situation `s1 < s2` is just a roundabout test for "s1 is the negative one," and either branch ends with both values positive. The rule collapses to: *whenever the two modes disagree, don't flip anything.*

That's a determinacy bug. Both $s^{(1)}$ and $s^{(2)}$ are odd functions of the component's current orientation — replacing $(u_k, v_k)$ with $(-u_k, -v_k)$ sends $\sum_j \mathrm{sign}(a_j)a_j^2 \mapsto -\sum_j \mathrm{sign}(a_j)a_j^2$ on both sides. So "leave it alone" preserves whatever arbitrary orientation LAPACK handed you, which is the thing the algorithm exists to remove. Here's a rank-1 example built so the modes disagree (positive-skewed scores, negative-skewed loadings), run twice on the two orientations LAPACK could have returned:

```
loadings-side s1 =  2262.09
scores-side   s2 =  -661.98

with abs: net orientation for the two input signs = [1.0,  1.0]   deterministic
no abs  : net orientation for the two input signs = [1.0, -1.0]   not deterministic
```

The magnitude comparison fixes this because $|s^{(1)}|$ and $|s^{(2)}|$ are *even* under an input flip. The winner is chosen orientation-independently, and its sign then flips with the input, so the net orientation is pinned.

It also matches the paper's stated rationale, which is about strength of evidence: the mode whose projections are larger has more to say about where the component points, so the weaker mode is the one that gets overruled. Comparing raw signed values doesn't measure evidence strength at all.

The $N$-way case makes it sharper. There the constraint is $\prod_n \mathrm{sign}(s^{(n)}_k) = +1$, so you flip one mode whenever the count of negatives is odd. With three modes all negative, `min(s)` without `abs` selects the *most* negative — the strongest evidence — and overrules exactly the mode you'd least want to overrule. With `abs` it selects the weakest, which is the intended behavior.

Practically, this branch is rarely hit: it needs the loading vector and the score vector to have third moments of opposite sign, which is unusual for a well-determined component and easy to miss in testing. If the paper's typeset pseudocode really does omit the bars, that's the most likely explanation for how it survived. Worth checking whether the accompanying MATLAB code from the Sandia report (SAND2007-6422) has them — if it does, it's a typesetting slip.

Q: if using tensorly or other tensor oriented library, how is the sign resolved

Short answer: none of the Python tensor libraries implement Bro–Acar–Kolda. TensorLy gives you two weaker mechanisms, and they solve different problems from each other.

First, a correction to what I wrote earlier — the paper's actual title is *Resolving the Sign Ambiguity in the Singular Value Decomposition*, Journal of Chemometrics 22(2):135–140, 2008. I gave you a garbled title.

**TensorLy: the SVD-level flip.** `tensorly.tenalg.svd_interface` takes `flip_sign=True` and `u_based_flip_sign=True`, described as resolving the sign indeterminacy of the SVD and choosing whether to resolve it using U or V. The rule is the max-abs one: the sign ambiguity is resolved by making the largest component in the columns of U positive — identical to sklearn's `svd_flip`. This fires during `init='svd'` for CP and Tucker, so it makes the *initialization* reproducible. It says nothing about the orientation of the converged factors.

**TensorLy: `cp_tensor.cp_flip_sign`.** This is a parking convention, not an evidence-based one. The default summarizing function is the mean; the loop skips the target mode, computes `sign(func(factors[jj], axis=0))` for every other mode, and multiplies that sign into both that factor and the target mode, so all negative signs end up assigned to the mode indicated by `mode`. Two consequences worth knowing. The non-target modes are oriented by the mean of their loadings — the "sum of loadings positive" rule, which is exactly the one that fails on contrast-like components whose entries cancel. And the target mode's sign isn't determined by anything; it just absorbs the parity residue.

`parafac` itself does not call any sign fix on the way out, so unless you invoke `cp_flip_sign` yourself, the signs are whatever ALS happened to land on.

**The structural difference from PCA.** For an $N$-way CP model with $\lambda_k \ge 0$, the sign freedom on component $k$ is $\{s \in \{\pm 1\}^N : \prod_n s_n = +1\}$ — that's $2^{N-1}$ equivalent patterns, not 2. A parking convention fixes $N-1$ of the modes by fiat and dumps the rest on one mode. BAK's actual content is that it scores *all* $N$ modes against the data, then repairs parity by overruling whichever mode has the weakest $|s^{(n)}_k|$ — the same tie-break we were just discussing, which is why the `abs` matters more here than in the two-way case. Also normalize before flipping (`cp_normalize`), or sign can hide in the weights instead of the factors.

**Where BAK actually lives.** The MATLAB Tensor Toolbox: `ktensor/fixsigns` fixes the sign ambiguity of a ktensor, in both the intrinsic form and a two-argument form that aligns to a reference ktensor. Note that the call to `fixsigns` inside `cp_als` was made optional, so don't assume it ran. There's a direct NumPy port in LANL's pyCP_APR: `pyCP_APR.numpy_backend.fixsigns_ktensor.fixsigns_oneargin(K)`, which fixes the sign ambiguity of a ktensor and returns it with the latent factor columns flipped. Julia has it too — TensorToolbox.jl exports `fixsigns` and `fixsigns!`.

**tensortools takes the other route.** Rather than an intrinsic convention it aligns fits to each other: `tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)` aligns two fits and returns a similarity score, on the premise that repeated CP fits recover the same result up to permutation and sign-flipping. Alex Williams wrote it for exactly the neural-data ensemble case.

For your EEG and MRI work I'd lean on the tensortools primitive rather than any intrinsic rule. Across subjects, sessions, or bootstrap replicates you have a permutation ambiguity too, and BAK doesn't touch that — you need Hungarian matching on $|\cos|$ similarity to pair components, with the sign falling out of the matched cosine. Use BAK once at the end, on the reference fit, so the published figure has a canonical orientation.

If you want the cheap middle path in TensorLy, `cp_flip_sign` accepts a custom `func`, so passing `lambda A, axis: np.sum(np.sign(A) * A**2, axis=axis)` gets you the signed-sum-of-squares statistic instead of the mean. That's BAK-flavored and strictly better than the mean, but it scores the factor columns rather than their projections onto the data, and it keeps the parking structure — so it isn't the algorithm.
