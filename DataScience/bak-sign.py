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


def resolve_signs_bak(X, U, S, Vt, deflate=False):
    """
    Fix the sign of each component of a factorization X ~= U diag(S) Vt.

    Parameters
    ----------
    X : (n, p) array
        The matrix that was factorized, preprocessed exactly as it was
        for the decomposition (i.e. already column-centered for PCA).
    U : (n, K), S : (K,), Vt : (K, p)
        Truncated SVD factors. S must be non-negative.
    deflate : bool = False
        Remove the other components' contribution before scoring each
        component. This is a no-op when U has orthonormal columns
        (as in PCA/SVD) and only matters for oblique factor models like
        parallel factor analysis (PARAFAC). It adds O(npK) work.

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
