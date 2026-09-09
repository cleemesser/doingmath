# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
#     "multimethod==1.10", # need to pin at 1.10 because v2 breaks scikit-fda
#     "scikit-fda==0.10.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    # /// script
    # python-version
    # dependencies = [
    #    "marimo",
    #    "numpy",
    #    "multimethod==1.10"
    #    "scikit-fda"
    #  ]
    pass


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understand "functional data analysis" and specifically functional PCA

    My impression is that this is a data reduction approach for dealing with features that are overall patterns in something like a time series.

    The issue is that these are very high (potentially infinite) dimensional objects (e.g. functions in a Hilbert space like $L^2(\mathbb{R}))$ so you often want some way to deal with this in a more tractable way, especially in the context of noise and intrinsic variation.

    My assumption is that you imagine data is being generated over time by a process $X(t)$. You get data samples $x_i(\tau_j)$ where for the $i$-th data point is sampled at times $\tau_j$ of fixed length $N$ which can be a large number.

    Let's get set to do this by creating a centered version of the data we will compute the mean waveform
    $$X^{c}_{it} = x_i(\tau_t) - \bar{x}_t$$
    where we will abuse notation a bit and use $t$ as a discrete index into the times $\tau_t$

    ## Functional principle components analysis seems like it is just regular principle components analysis (PCA)
    If that is so, then we should just be able to perform a SVD on this matrix and get our principle components and eigenvalues to asses for how much variance each component accounts for.

    We will try this below and then compare with the output of the scikit-fda package (scikit functional data analysis).

    #### Mathematical Note
    PCA is often defined by first formming the covariance matrix, $C$, which for the mean zero data matrix $X$.
    $$ C = \operatorname{average}( X^T X )= \frac{X^T X}{n-1} $$
    and then finding the eigenvectors and eigenvalues of $C$. These are the principal components.
    with SVD
    $$ X^{c} = U S V^T $$

    where if $X^{c}$ is mxn, $U$ is a mxm orthogonal matrix (a rotation), $S$ is
    the scaling diagonal matrix of "singular values" in decreasing order. And
    $V$ is the nxn orthogonal matrix which is another rotation. For complex
    matrices, substitute unitary for orthogonal and conjugate transpose for the
    regular adjoint operation See for example, [wikipedia
    SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)

    $$ \frac{1}{n-1}(U S V^T)^T (U S V^T) = \frac{1}{n-1}V S U^T U S V^T $$

    $$ U^T U = I $$
    so the above equals:


    $$  V \frac{S^2}{n-1} V^T $$

    which is the solution to the eigvectors for $C$ where the columns (and rows)
    of $V$ are the eigenvectors of $C$ and diagnonal $$ are the eigenvalues.
    """)
    return


@app.cell
def _():
    import numpy as np
    from typing import Literal
    import skfda
    import matplotlib.pyplot as plt  # maybe use plotly instead

    return Literal, np, plt, skfda


@app.cell(hide_code=True)
def _(skfda):
    Xg, yg = skfda.datasets.fetch_growth(return_X_y=True)
    return (Xg,)


@app.cell(hide_code=True)
def _(Xg):
    Xg.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### The Berkeley growth dataset
    We have imported skfda and imported the Berkley Growth dataset. This has 93 children as subjects with their height measured in centimeters at 31 times as described below by Xg.grid_points (starts at age 1.0, then 1.25, ...). I will grab the growth data as a numpy array and remove the unused dimension to create a 2D matrix in Xarr, while storing the time grid points in Tarr (shown below):
    """)
    return


@app.cell
def _(Xg):
    Xarr = Xg.data_matrix.squeeze()
    # Xarr.shape -> (93,31)
    Tarr = Xg.grid_points[0].squeeze()  # timepoints
    # Tarr.shape -> (31,): 1,1.25, 1.5, 1.75 ...
    Tarr
    return Tarr, Xarr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So now to do PCA on this data, I'm going to use `numpy.linalg.svd` but I am also going to add a helper function to deal with an ambiguity in SVD (or any eigenvector problem): the eigenvectors define a direction but it does not matter whether the vector or its negative is used. I will define a helper function `sign_resolved_svd()`.
    """)
    return


@app.cell
def _(Literal, n_modes, np):
    # define "sign resolved" SVD using eeither Bro–Acar–Kolda or sklearn svd sign convention here
    # this is not absolutely necessary but is a way to get consistent results
    # otherwise just use np.linalg.svd

    def sign_resolved_svd(
        X,
        method: Literal["max_ucoef_pos", "bak"] = "max_ucoef_pos",
        n_components=None,
        deflate=True,
    ):
        """
        Compute SVD with numpy.linalg.svd
        but return vectors with  sign ambiguity resolved via one of two      conventions.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input matrix
        n_components : int, optional
            Number of components (modes) to compute (default: all)
        method: str defaults to max_coeef_pos because it is prob faster
            'max_coeff_pos' - align vectors so the max coeff is positive
            'bak' - use Bro,Akar,Kolda data-aligned convention
        deflate : bool
            Remove the other components' contribution before scoring each
            component. This is a no-op when U has orthonormal columns
            (as in PCA/SVD) and only matters for oblique factor models.

        Returns:
        --------
        U : array, left singular vectors
        s : array, singular values
        Vt : array, right singular vectors (transposed)

        TODO: add vcoef based version of method
        """
        # Compute standard SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        if n_components is not None:
            U = U[:, :n_modes]
            s = s[:n_modes]
            Vt = Vt[:n_modes, :]

        # overwrite the old to release memory, toss flips
        if method == "max_ucoef_pos":
            U, Vt = svd_max_u_flip(U, Vt)
        elif method == "bak":
            U, Vt = svd_bak_sign_flip(X, U, s, Vt,deflate=deflate)
        else:
            raise (Exception("unrecognized method option {e}"))

        return U, s, Vt

    def svd_max_u_flip(U, Vt):
        """Sign correction to ensure deterministic output from SVD.
        from sklearn: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py 2026-08-26
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive. More flexible version in sklearn allows you to chose v instead of u.
        X = U @ np.diag(s) @ Vt    svd decomposition Vt <-> V.T


        Parameters
        ----------
        u : ndarray
            Parameters u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
            u can be None if `u_based_decision` is False.

        v : ndarray
            Parameters u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(U * s, Vt)`. The input v should
            really be called vt to be consistent with scipy's output.
            v can be None if `u_based_decision` is True.

        Returns
        -------
        u_adjusted : ndarray
            Array u with adjusted columns and the same dimensions as u.

        v_adjusted : ndarray
            Array v with adjusted rows and the same dimensions as v.
        """

        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = np.argmax(np.abs(U.T), axis=1)
        shift = np.arange(U.T.shape[0])
        indices = max_abs_u_cols + shift * U.T.shape[1]
        signs = np.sign(np.take(np.reshape(U.T, (-1,)), indices, axis=0))
        U *= signs[np.newaxis, :]
        if Vt is not None:
            Vt *= signs[:, np.newaxis]

        return U, Vt

    def svd_bak_sign_flip(X, U, s, Vt) -> (np.ndarray, np.ndarray):
        """
        correct signs of SVD following BAK data oriented convention

        where U,s,Vt = numpy.linalg.svd(X)
        this one uses the data to chose the sign of the singular vectors
        lower case for 1d arrays, upper case for 2d arrays
        slightly modified from David Willo gist:
        https://gist.github.com/David-Willo/1825bf9e8c30e13147e332734bcaebd5

        Bro, R., Acar, E., & Kolda, T. G. (2008). Resolving the sign ambiguity in the singular value decomposition.
        Journal of Chemometrics: A Journal of the Chemometrics Society, 22(2), 135-140.
        URL: https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf
        note the pseudo-code seems to has an error for s_right and s_left use
        it is definitely different between the report and the paper.
        """
        # SDV dimensions:
        # U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # X = U @ diag(S) @ Vt
        # (I,J) = (I,K) @ (K,K) @ (K,J)

        # U, S, Vt = np.linalg.svd(X, full_matrices=False)

        I = U.shape[0]
        J = Vt.shape[1]
        K = s.shape[0]

        assert U.shape == (I, K)
        assert Vt.shape == (K, J)
        assert X.shape == (I, J)
        # if don't want to alter originals
        U = U.copy()
        Vt = Vt.copy()

        s_left, s_right = np.zeros(K), np.zeros(K)

        for k in range(K):
            mask = np.ones(K).astype(bool)
            mask[k] = False
            # (I,J) = (I,K-1) @ (K-1,K-1) @ (K-1,J)
            Y = X - (U[:, mask] @ np.diag(s[mask]) @ Vt[mask, :])

            for j in range(J):
                d = np.dot(U[:, k], Y[:, j])
                s_left[k] += np.sum(np.sign(d) * d**2)
            for i in range(I):
                d = np.dot(Vt[k, :], Y[i, :])
                s_right[k] += np.sum(np.sign(d) * d**2)

        #print("about to calculate BAK flips ;-)")
        for k in range(K):
            if (s_left[k] * s_right[k]) < 0:  # one of them is negative
                # indicate flip the one whose abs magnittude is smaller
                if np.abs(s_left[k]) < np.abs(s_right[k]):
                    s_left[k] = -s_left[k]
                else:
                    s_right[k] = -s_right[k]
            U[:, k] = U[:, k] * np.sign(s_left[k])
            Vt[k, :] = Vt[k, :] * np.sign(s_right[k])

        return U, Vt

    return (sign_resolved_svd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculating the PCA with SVD

    You can do PCA by finding the SVD composition on the centered data $X^c$:
    $$
    (U, s, V^T) = \operatorname{sign\_resolved\_svd}(X^c)
    $$

    so
    $$ X^c = U S V^T $$
    Doing this in code and then plotting the singular values below, we can see some things.
    """)
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(Xarr, np, plt, sign_resolved_svd):
    Xarr_m = Xarr.mean(axis=0)
    Xarr_c = Xarr - Xarr_m
    U, s, Vt = sign_resolved_svd(Xarr_c, method='max_ucoef_pos',
                                 #method="bak"
                                )
    print(f'{Xarr.shape=}')
    var = s**2 / (Xarr.shape[0] - 1) # var = S**2 /(n_subjects-1)
    var_percent = var / var.sum()

    # can plot a lot of things to show similar information
    #plt.plot(s, label="singular values")
    # plt.yscale('log') # falls really fast so use log scale
    #plt.title("singular values")
    # plt.plot(var, label="variances")
    color_sv = 'tab:red'

    _fig,ax = plt.subplots(1,2)
    ax[0].plot(s, '-x',label="singular values", color=color_sv)
    ax[0].set_ylabel('singular values')
    ax[0].set_xlabel('component number')
    ax[0].tick_params(axis='y', labelcolor=color_sv)
    ax[0].legend()
    #ax2=ax[0].twinx()
    color2 = 'blue'
    color3 = 'lightblue'
    ax[1].plot(100 * var_percent, "-+", color=color2, label="%variance per PC")
    ax[1].plot(np.cumsum(100 * var_percent), '-|',color=color3, label="cumulative var.")
    ax[1].set_title('Percent variance accounted for per PC')
    #plt.title("percent variance accounted for by PC")


    ax[1].legend()
    # plt.yscale('log')
    _fig.tight_layout()
    _fig
    return U, Vt, Xarr_m, s, var_percent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From these graphs we conclude that virtually all the variance in the data is accounted for within the first 3 or 4 principle components.

    We can reconstruct a data point using as many components as we want. For example, we can calculate the reconstructed $\hat{x}_i$ with two components by projecting a given subject growth profile $x_i$ onto singular vector/principle component $V_j$, which is a column or row of $V^T$, via the inner product and adding it back to the mean vector:
    $$ \hat{x}_i = \bar{x} + <x_i, V_0> V_0 + <x_i, V_1> V_1 $$

    We will do that below in code and then plot the original growth curve and the reconstructed growth curve using two components.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive widget
    You can play with this yourself by using the sliders to select a subject and how many components to use for the reconstruction below.
    """)
    return


@app.cell(hide_code=True)
def _(Tarr, Vt, Xarr, Xarr_m, plt):

    x1 = Xarr_m + Xarr[0]
    x1_recon = Xarr_m + (Xarr[0].T @ Vt[0])* Vt[0] + (Xarr[0] @ Vt[1]) * Vt[1]

    plt.plot(Tarr, x1, label="original")
    plt.plot(Tarr, x1_recon, label="reconstructed from 2 PC")
    plt.legend()
    plt.title("reconstructed vs original waveform")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive widget
    You can play with this yourself by using the sliders to select a subject and how many components to use for the reconstruction below.
    """)
    return


@app.cell(hide_code=True)
def _(U, Vt, Xarr, Xarr_m, np, s, slider):
    # now do full recon
    n_pcs = slider.value # need to be in separate cell from slider below
    s_recon = s.copy()
    s_recon[n_pcs:] = 0.0
    Xarr_c_recon = U @ np.diag(s_recon) @ Vt
    Xarr_c_recon.shape
    Xarr_recon = Xarr_m + Xarr_c_recon
    recon_error = np.sqrt(np.sum((Xarr_recon - Xarr) ** 2))
    return Xarr_recon, n_pcs


@app.cell
def _(mo):
    slider = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=3,
        label="Number of principle components",
    )

    slider_subject = mo.ui.slider(
        start=0, stop=90, step=1, value=1, label="Subject No"
    )
    mo.vstack([slider_subject, slider])
    return slider, slider_subject


@app.cell(hide_code=True)
def _(Tarr, Xarr, Xarr_recon, n_pcs, plt, slider_subject):
    _i = slider_subject.value
    plt.plot(Tarr, Xarr_recon[_i], label=f"recon with {n_pcs} PCs")
    plt.plot(Tarr, Xarr[_i], label="original")
    plt.legend()
    plt.title(f"Reconstructed and original growth curve for subject {_i}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taking a look at the shape of the principal components
    Let's plot out the first few components that capture most of the variance in the data
    """)
    return


@app.cell
def _(Tarr, Vt, Xarr_m, plt):
    # probably should plot out the first 3 or 4 PCs here
    _Npcs = 3
    _fig, _axs = plt.subplots(1, 2)
    # probably should plot out the first 3 PCs here
    for _ii in range(_Npcs):
        _axs[0].plot(Tarr, Vt[_ii], label=f"PC{_ii}")
    _axs[0].set_title(f"first {_Npcs} principal components")
    _axs[0].legend()

    for _ii in range(_Npcs):
        _axs[1].plot(
            Tarr, 20 * Vt[_ii] + Xarr_m, label=f"PC{_ii}+" r"$\bar{x}$"
        )
    _axs[1].set_ylabel('height (cm)')
    _axs[1].set_xlabel('age (years)')
    _axs[1].plot(Tarr, Xarr_m, ".", label=r"$\bar{x}$ (mean)")
    _axs[1].legend()
    _axs[1].set_title(f"scaled first {_Npcs} PCs + mean vector")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The graph on the right side is made by scaling the principal components, that is, multiplying them by 20 cm, and then adding them back to the mean curve.

    As a pediatrician, this looks a lot like the first component corresponds to the boys growth curve and the second one curresponds to girls growth curve and the third one is able mediate differences between them. Girls start to grow at little earlier  than boys, but also hit their maximum height earlier. Boys start a little later, but continue to grow longer on average and are, on average, a bit taller.

    ### Comparison with the scikit-fda package for functional data analysis

    Next I found the functional PCA in skfda and ran its "fit" function and compared it with our analysis above
    """)
    return


@app.cell
def _():
    import skfda.preprocessing.dim_reduction as dim_reduction

    return (dim_reduction,)


@app.cell
def _(dim_reduction):
    # this will break if you don't pin multimethod==1.10 (vs 2.x)
    fPCA = dim_reduction.FPCA(3)
    return (fPCA,)


@app.cell
def _(Xg, fPCA):
    fPCA.fit(Xg)
    return


@app.cell
def _(Xarr_m, fPCA, np, var_percent):
    # are the X mean vectors the same?
    np.allclose(fPCA.mean_.data_matrix, Xarr_m)
    var_percent[
        :4
    ]  # array([0.80362926, 0.14055602, 0.02757923, 0.01456779]) matches
    return


@app.cell
def _(fPCA, plt):
    fPCA.components_.plot()
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These qualitatively match with the green looking like the negative of PC[2]
    """)
    return


@app.cell
def _(Vt, Xarr, fPCA, mo, np):
    # import marimo as mo
    # import numpy as np
    # import skfda
    # import skfda.preprocessing.dim_reduction as dim_reduction

    # /// script
    # dependencies = ["marimo", "numpy", "scikit-fda"]

    # Xg, _ = skfda.datasets.fetch_growth(return_X_y=True)
    # Xarr = Xg.data_matrix.squeeze()
    # Xarr_m = Xarr.mean(axis=0)
    # Xarr_c = Xarr - Xarr_m

    # Your BAK-convention SVD # calculated above
    # U, s, Vt = sign_resolved_svd(Xarr_c, method="bak")

    # scikit-fda's functional PCA # also calculated above
    # fPCA = dim_reduction.FPCA(3)
    # fPCA.fit(Xg)

    # fPCA component curves (k, n_times) — align sign to your Vt
    _msg_list = []
    _n_to_check = 3
    _corr_arr = np.zeros((3,), dtype=float)
    for k in range(_n_to_check):
        fpc = (
            np.asarray(fPCA.components_[k].data_matrix)
            .reshape(-1, Xarr.shape[1])
            .ravel()
        )
        corr = np.corrcoef(Vt[k], fpc)[0, 1]
        _corr_arr[k] = corr
        _msg_list.append(
            f"PC {k}: corr = {corr:+.4f}  (|corr|≈1 means same curve, sign only differs)\n"
        )

    _eps = 0.025
    _the_same = np.allclose(np.abs(_corr_arr), np.ones(_corr_arr.shape),rtol=_eps)
    # _the_same = np.abs(_corr_arr)
    mo.md("- ".join(_msg_list)  +     f"all the absolute cross correlation are within {_eps} of 1.0" if _the_same else None )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So this verifies that my code and the skfda package are doing basically the same thing.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
