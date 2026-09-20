# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
#     "multimethod==1.10", # need to pin at 1.10 because v2 breaks scikit-fda
#     "scikit-fda==0.10.1",
#     "tensorly",
#     "marimo>=0.24.1",
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
    #    "scikit-fda",
    #  ]
    pass


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parallel Factor Analysis
    - PCA only considers orthogonal components
    - try the growth set again, look at other sets
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a 2D matrix, PARAFAC decomposes $X^c \approx \sum_{r=1}^R a_r \otimes b_r$ — a sum of $R$ rank-1 terms. Unlike SVD, the factor vectors ${a_r}$ and ${b_r}$ are not constrained to be orthogonal, which is the key distinction your notebook motivates.
    """)
    return


@app.cell
def _():
    import numpy as np
    from typing import Literal
    import skfda
    import matplotlib.pyplot as plt  # maybe use plotly instead
    import tensorly as tl
    from tensorly.decomposition import parafac

    return np, parafac, plt, skfda, tl


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
def _():
    return


@app.cell
def _(Xarr, np, parafac, tl):
    # import tensorly is already present

    Xarr_c = Xarr - Xarr.mean(axis=0)

    # Sweep the number of PARAFAC factors
    ranks = list(range(1, 11))
    parafac_recon = {}  # rank -> reconstructed matrix
    parafac_error = []  # normalised Frobenius error

    for _r in ranks:
        cp_tensor = parafac(
            Xarr_c,
            rank=_r,
            n_iter_max=500,
            init="svd",
            random_state=42,
            tol=1e-6,
        )
        X_recon = tl.cp_to_tensor(cp_tensor)
        parafac_recon[_r] = X_recon

        # relative reconstruction error (||X_c - X_recon||_F / ||X_c||_F)
        err = np.linalg.norm(Xarr_c - X_recon) / np.linalg.norm(Xarr_c)
        parafac_error.append(err)

    print(f"{'Rank':>5} | {'Rel. Error':>10} | {'% Var Explained':>16}")
    print("-" * 38)
    for _r, _e in zip(ranks, parafac_error):
        print(f"{_r:>5} | {_e:>10.6f} | {(1 - _e**2) * 100:>15.2f}%")
    return Xarr_c, parafac_error, parafac_recon, ranks


@app.cell
def _():
    return


@app.cell
def _(parafac_error, plt, ranks):
    _fig_p, _ax = plt.subplots(figsize=(7, 4))
    _ax.bar(
        ranks,
        [(1 - e**2) * 100 for e in parafac_error],
        color="steelblue",
        edgecolor="navy",
        alpha=0.85,
        label="PARAFAC",
    )
    _ax.set_xlabel("Number of PARAFAC factors (R)")
    _ax.set_ylabel("Cumulative % variance explained")
    _ax.set_title("PARAFAC reconstruction quality vs. number of factors")
    _ax.set_xticks(ranks)
    _ax.set_ylim(0, 102)
    for _i, (_rv, _ev) in enumerate(zip(ranks, parafac_error)):
        _ax.text(
            _rv,
            (1 - _ev**2) * 100 + 1.2,
            f"{(1 - _ev**2) * 100:.1f}",
            ha="center",
            fontsize=8,
        )
    _ax.legend()
    _fig_p.tight_layout()
    _fig_p
    return


@app.cell
def _(Tarr, Xarr, parafac_recon, plt):
    # Pick a few representative ranks to overlay on the original curve
    _selected_ranks = [1, 2, 3, 5, 8]
    _subject_idx = 14  # pick one subject to inspect

    _fig3, _axs3 = plt.subplots(
        1, len(_selected_ranks), figsize=(16, 3.4), sharey=True
    )
    for _ax, _r in zip(_axs3, _selected_ranks):
        _ax.plot(Tarr, Xarr[_subject_idx], "k-", lw=1.8, label="original")
        _ax.plot(
            Tarr,
            parafac_recon[_r][_subject_idx] + Xarr.mean(axis=0),
            "r--",
            lw=1.4,
            label=f"PARAFAC R={_r}",
        )
        _ax.set_title(f"{_r} factor{'s' if _r > 1 else ''}")
        _ax.set_xlabel("age (years)")
        _ax.legend(fontsize=8)
    _axs3[0].set_ylabel("height (cm)")
    _fig3.suptitle(
        f"Reconstruction of subject {_subject_idx} at different factor counts",
        y=1.02,
    )
    _fig3.tight_layout()
    _fig3
    return


@app.cell
def _(mo):
    r_slider = mo.ui.slider(
        start=1, stop=10, step=1, value=3, label="PARAFAC factors"
    )
    s_slider = mo.ui.slider(start=0, stop=92, step=1, value=0, label="Subject")
    mo.hstack([s_slider, r_slider])
    return r_slider, s_slider


@app.cell
def _(Tarr, Xarr, parafac_error, parafac_recon, plt, r_slider, s_slider):
    _sel_subj = s_slider.value
    _sel_rank = r_slider.value
    _err = parafac_error[_sel_rank - 1]

    _fig_i = plt.figure(figsize=(7, 4))
    plt.plot(Tarr, Xarr[_sel_subj], "k-", lw=2, label="original")
    plt.plot(
        Tarr,
        parafac_recon[_sel_rank][_sel_subj] + Xarr.mean(axis=0),
        "r--",
        lw=1.6,
        label=f"PARAFAC R={_sel_rank}",
    )
    plt.legend()
    plt.xlabel("age (years)")
    plt.ylabel("height (cm)")
    plt.title(
        f"Subject {_sel_subj} — R={_sel_rank}, "
        f"var explained = {(1 - _err**2) * 100:.1f}%"
    )
    plt.gca()
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
    x1_recon = Xarr_m + (Xarr[0].T @ Vt[0]) * Vt[0] + (Xarr[0] @ Vt[1]) * Vt[1]

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
    n_pcs = slider.value  # need to be in separate cell from slider below
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
    _axs[1].set_ylabel("height (cm)")
    _axs[1].set_xlabel("age (years)")
    _axs[1].plot(Tarr, Xarr_m, ".", label=r"$\bar{x}$ (mean)")
    _axs[1].legend()
    _axs[1].set_title(f"scaled first {_Npcs} PCs + mean vector")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(Xarr_c, parafac):
    # Extract the factor matrices from a rank-4 PARAFAC decomposition
    _Nf = 3
    weights4display, factors4display = parafac(
        Xarr_c, rank=_Nf, n_iter_max=500, init="svd", random_state=42, tol=1e-6
    )

    # factors4 is a list: [mode-0 (subjects), mode-1 (time)]
    subject_factors = factors4display[0]  # shape (93, 4)
    time_factors = factors4display[
        1
    ]  # shape (31, 4)  ← these are the "shape" curves
    return (time_factors,)


@app.cell
def _(Tarr, Xarr, plt, time_factors):
    _Nf = 3
    _fig_f, _axs_f = plt.subplots(2, 1, figsize=(4.5, 9.0))

    # Left: raw time-profile factors
    for _k in range(_Nf):
        _axs_f[0].plot(Tarr, time_factors[:, _k], label=f"Factor {_k + 1}")
    _axs_f[0].set_xlabel("age (years)")
    _axs_f[0].set_ylabel("loading")
    _axs_f[0].set_title(f"First {_Nf} PARAFAC time-profiles (raw)")
    _axs_f[0].legend()
    _axs_f[0].axhline(0, color="grey", lw=0.6, ls="--")

    # Right: scaled factors offset by the mean curve (like the PC plot you made)
    _Xmean = Xarr.mean(axis=0)
    for _k in range(_Nf):
        _axs_f[1].plot(
            Tarr, 20 * time_factors[:, _k] + _Xmean, label=f"Factor {_k + 1}"
        )
    _axs_f[1].plot(Tarr, _Xmean, ".", color="black", ms=4, label=r"$\bar{x}$")
    _axs_f[1].set_xlabel("age (years)")
    _axs_f[1].set_ylabel("height (cm)")
    _axs_f[1].set_title(f"scaled PARAFAC factors + mean")
    _axs_f[1].legend()
    _fig_f.tight_layout()
    _fig_f
    return


@app.cell
def _(skfda):
    # Fetch the 'growth' dataset from the R 'fda' package which includes both height and weight
    # Note: You may need the 'scikit-datasets' package installed under the hood
    # zurich_growth = skfda.datasets.fetch_cran("growth","fda")
    # print(.shape)  # expect (n_subj, n_times, n_vars)

    children_fda = skfda.datasets.fetch_cran("children", "npregfast")
    print(f"Loaded via skfda.fetch_cran ✓, returning {type(children_fda)}")
    children_fda  # dict I'm leaving this here because I think this is an awesome dataview
    return (children_fda,)


@app.cell
def _(children_fda, np):
    children_data = np.asarray(children_fda.data_matrix)
    children_times = children_fda.grid_points[0]
    print(f"Shape: {children_data.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The graph on the right side is made by scaling the principal components, that is, multiplying them by 20 cm, and then adding them back to the mean curve.

    As a pediatrician, this looks a lot like the first component corresponds to the boys growth curve and the second one curresponds to girls growth curve and the third one is able mediate differences between them. Girls start to grow at little earlier  than boys, but also hit their maximum height earlier. Boys start a little later, but continue to grow longer on average and are, on average, a bit taller.

    ## Futher compression - using a basis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These qualitatively match with the green looking like the negative of PC[2]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So this verifies that my code and the skfda package are doing basically the same thing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parallel Factors applied to handwriting dataset

    [data set source cran](https://rdrr.io/cran/fda/man/handwrit.html)
    Data representing the X-Y coordinates along time obtained while writing the word “fda”. The sample contains 20 instances measured over 2.3 seconds that had been aligned for a better understanding. Each instance is formed by 1401 coordinate values Ramsay and Silverman (chapter 3)

    These data are the X-Y coordinates of 20 replications of writing the script "fda". The subject was Jim Ramsay. Each replication is represented by 1401 coordinate values. The scripts have been extensively pre-processed. They have been adjusted to a common length that corresponds to 2.3 seconds or 2300 milliseconds, and they have already been registered so that important features in each script are aligned.

    This analysis is designed to illustrate techniques for working with functional data having rather high frequency variation and represented by thousands of data points per record. Comments along the way explain the choices of analysis that were made.

    The final result of the analysis is a third order linear differential equation for each coordinate forced by a constant and by time. The equations are able to reconstruct the scripts to a fairly high level of accuracy, and are also able to accommodate a substantial amount of the variation in the observed scripts across replications. by contrast, a second order equation was found to be completely inadequate.

    An interesting surprise in the results is the role placed by a 120 millisecond cycle such that sharp features such as cusps correspond closely to this period. This 110-120 msec cycle seems is usually seen in human movement data involving rapid movements, such as speech, juggling and so on.

    These 20 records have already been normalized to a common time interval of 2300 milliseconds and have beeen also registered so that prominent features occur at the same times across replications. Time will be measured in (approximate) milliseconds and space in meters. The data will require a small amount of smoothing, since an error of 0.5 mm is characteristic of the OPTOTRAK 3D measurement system used to collect the data.

    Milliseconds were chosen as a time scale in order to make the ratio of the time unit to the inter-knot interval not too far from one. Otherwise, smoothing parameter values may be extremely small or extremely large.

    The basis functions will be B-splines, with a spline placed at each knot. One may question whether so many basis functions are required, but this decision is found to be essential for stable derivative estimation up to the third order at and near the boundaries.
    """)
    return


@app.cell
def _(skfda):
    handwriting_data = skfda.datasets.fetch_handwriting(return_X_y=True)
    hw, hw_y = handwriting_data
    handwriting_data
    return (hw,)


@app.cell
def _(hw, np):
    # data_matrix: (n_samples, n_time, n_coords)
    #   n_samples = 750 (75 writers × 10 digits)
    #   n_coords  = 2   (x, y penposition)
    Xhw_arr = np.asarray(hw.data_matrix)  # (20, 1401, 2)
    return (Xhw_arr,)


@app.cell
def _(Xhw_arr):
    Xhw_arr[0, 0:10, :]
    return


@app.cell
def _(hw):
    hw.grid_points[0]  # times in milliseconds
    return


@app.cell
def _():
    return


@app.cell
def _(Xhw_arr):
    # Centre each variable over samples
    hw_mean = Xhw_arr.mean(axis=0, keepdims=True)  # (1, T, 2)
    hw_c = Xhw_arr - hw_mean
    return hw_c, hw_mean


@app.cell
def _(hw_c, np, parafac, tl):
    _RandSeed = 42
    hw_norm = np.linalg.norm(hw_c)

    # Sweep PARAFAC rank
    hw_ranks = list(range(1, 11))
    hw_parafac_recon = {}
    hw_parafac_error = []

    for _r in hw_ranks:
        _cp = parafac(
            hw_c,
            rank=_r,
            n_iter_max=2000,
            init="svd",
            random_state=_RandSeed,
            tol=1e-8,
        )
        _rec = tl.cp_to_tensor(_cp)
        hw_parafac_recon[_r] = _rec
        hw_parafac_error.append(np.linalg.norm(hw_c - _rec) / hw_norm)

    print(f"{'Rank':>5} | {'Rel. Error':>10} | {'% Var Explained':>16}")
    print("-" * 38)
    for _r, _e in zip(hw_ranks, hw_parafac_error):
        print(f"{_r:>5} | {_e:>10.6f} | {(1 - _e**2) * 100:>15.2f}%")
    return hw_parafac_error, hw_parafac_recon, hw_ranks


@app.cell
def _(hw_parafac_error, hw_ranks, plt):
    # Variance-explained bar chart
    _fig_hw, _ax_hw = plt.subplots(figsize=(7, 4))
    _ax_hw.bar(
        hw_ranks,
        [(1 - e**2) * 100 for e in hw_parafac_error],
        color="coral",
        edgecolor="firebrick",
        alpha=0.9,
    )
    _ax_hw.set_xlabel("Number of PARAFAC factors (R)")
    _ax_hw.set_ylabel("Cumulative % variance explained")
    _ax_hw.set_title("Handwriting — PARAFAC reconstruction quality")
    _ax_hw.set_xticks(hw_ranks)
    _ax_hw.set_ylim(0, 102)
    for _rv, _ev in zip(hw_ranks, hw_parafac_error):
        _ax_hw.text(
            _rv,
            (1 - _ev**2) * 100 + 1.2,
            f"{(1 - _ev**2) * 100:.1f}",
            ha="center",
            fontsize=8,
        )
    _fig_hw.tight_layout()
    _fig_hw
    return


@app.cell
def _(Xhw_arr, hw_mean, hw_parafac_recon, plt):
    # Reconstructed x and y curves for one writing sample
    _example_idx = 0  # example 0 out of 20
    _hw_sel_ranks = [2, 4, 6, 8]

    _fig_hwr, _axs_hwr = plt.subplots(1, 2, figsize=(13, 4.5))

    for _ax, _col, _lab in zip(_axs_hwr, [0, 1], ["x(t)", "y(t)"]):
        _ax.plot(Xhw_arr[_example_idx, :, :], "k-", lw=2, label="original")
        for _r, _c in zip(
            _hw_sel_ranks, ["steelblue", "darkorange", "seagreen", "purple"]
        ):
            _ax.plot(
                hw_parafac_recon[_r][_example_idx, :, :] + hw_mean[0, :, :],
                "--",
                lw=1.3,
                color=_c,
                label=f"R={_r}",
            )
        _ax.set_xlabel("time (normalised)")
        _ax.set_ylabel(_lab)
        _ax.set_title(f"{_lab} reconstruction — writing #{_example_idx}")
        _ax.legend(fontsize=8)

    _fig_hwr.suptitle(f'Pen position over time (20 reps of "fda")')
    _fig_hwr.tight_layout()
    _fig_hwr
    return


@app.cell
def _(Xhw_arr, plt):
    _example_idx = 0  # example 0 out of 20
    _hw_sel_ranks = [2, 4, 6, 8]

    _fig_hwr, _axs_hwr = plt.subplots(1, 2, figsize=(13, 4.5))
    _ax = _axs_hwr[0]
    # _x,_y = zip(*Xhw_arr[_example_idx,:,:].astype('float'))
    # _ax.scatter(_x,_y, "k-", lw=2, label="original")
    _x, _y = Xhw_arr[0, :, 0], Xhw_arr[0, :, 1]
    _x, _y
    _ax.plot(_x, _y, "k-", lw=2, label="original", alpha=0.3)
    return


@app.cell
def _(Xhw_arr, hw_mean, hw_parafac_recon, plt):
    _example_idx = 0  # example 0 out of 20
    _hw_sel_ranks = [2, 4, 6, 8]

    _fig_hwr, _axs_hwr = plt.subplots(1, 2, figsize=(6, 3.0))
    _ax = _axs_hwr[0]
    # _x,_y = zip(*Xhw_arr[_example_idx,:,:].astype('float'))
    # _ax.scatter(_x,_y, "k-", lw=2, label="original")
    for _example_idx in range(Xhw_arr.shape[0]):
        _x, _y = Xhw_arr[_example_idx, :, 0], Xhw_arr[_example_idx, :, 1]

        _ax.plot(_x, _y, "k-", lw=2, label="original", alpha=0.1)

    _ax = _axs_hwr[1]
    _r = 4
    for _example_idx in [2]:  # range(Xhw_arr.shape[0]):
        _tmp = hw_parafac_recon[_r][_example_idx, :, :] + hw_mean[0, :, :]
        _x, _y = _tmp[:, 0], _tmp[:, 1]
        _ax.plot(_x, _y, "r-", lw=2, label=f"recon rank:{_r}")
        _x, _y = Xhw_arr[_example_idx, :, 0], Xhw_arr[_example_idx, :, 1]
        _ax.plot(_x, _y, "k-", lw=2, label="original", alpha=0.3)

    _fig_hwr
    return


@app.cell
def _(hw, hw_c, np, parafac, plt):
    # First 6 PARAFAC factors — time profiles (mode 1) and variable loadings (mode 2)
    hw_Nf = 6
    _hw_cp = parafac(
        hw_c, rank=hw_Nf, n_iter_max=2000, init="svd", random_state=42, tol=1e-8
    )
    hw_weights, hw_factors = _hw_cp

    hw_time_factors = hw_factors[1]  # (T, 6)
    hw_var_factors = hw_factors[2]  # (2, 6) ← x/y loadings per factor

    _fig_hwf, _axs_hwf = plt.subplots(
        2, 1, figsize=(8, 8), height_ratios=[3, 1]
    )

    # --- time profiles ---
    for _k in range(hw_Nf):
        _axs_hwf[0].plot(
            hw.grid_points[0],
            hw_time_factors[:, _k],
            lw=1.5,
            label=f"Factor {_k + 1}",
        )
    _axs_hwf[0].set_xlabel("time (normalised)")
    _axs_hwf[0].set_ylabel("loading")
    _axs_hwf[0].set_title(f"First {hw_Nf} PARAFAC time-profiles")
    _axs_hwf[0].legend(loc="upper right", fontsize=8)
    _axs_hwf[0].axhline(0, color="grey", lw=0.6, ls="--")

    # --- variable-mode loadings (x vs y per factor) ---
    _xpos = np.arange(hw_Nf)
    _bw = 0.3
    _axs_hwf[1].bar(
        _xpos - _bw / 2,
        hw_var_factors[0, :],
        width=_bw,
        label="x-coord",
        color="steelblue",
    )
    _axs_hwf[1].bar(
        _xpos + _bw / 2,
        hw_var_factors[1, :],
        width=_bw,
        label="y-coord",
        color="coral",
    )
    _axs_hwf[1].set_xticks(_xpos)
    _axs_hwf[1].set_xticklabels([f"F{_k + 1}" for _k in range(hw_Nf)])
    _axs_hwf[1].axhline(0, color="grey", lw=0.6)
    _axs_hwf[1].set_ylabel("loading")
    _axs_hwf[1].set_title("Variable-mode loadings: x vs y per factor")
    _axs_hwf[1].legend()

    _fig_hwf.tight_layout()
    _fig_hwf
    return


@app.cell
def _(mo):
    # Interactive viewer: pick a digit and see the reconstruction in 2-D
    hw_r_sl = mo.ui.slider(1, 10, value=4, label="PARAFAC rank")
    hw_d_sl = mo.ui.slider(0, 9, value=3, label="Writing Sample")
    hw_s_sl = mo.ui.slider(0, 74, value=0, label="Subject")
    mo.hstack([hw_d_sl, hw_r_sl])
    return hw_d_sl, hw_r_sl, hw_s_sl


@app.cell(hide_code=True)
def _(
    Xhw_arr,
    hw_d_sl,
    hw_mean,
    hw_parafac_error,
    hw_parafac_recon,
    hw_r_sl,
    hw_s_sl,
    plt,
):
    _hw_i = hw_s_sl.value * 10 + hw_d_sl.value
    _hw_r = hw_r_sl.value
    _hw_err = hw_parafac_error[_hw_r - 1]

    _orig_x = Xhw_arr[_hw_i, :, 0]
    _orig_y = Xhw_arr[_hw_i, :, 1]
    _rec_x = hw_parafac_recon[_hw_r][_hw_i, :, 0] + hw_mean[0, :, 0]
    _rec_y = hw_parafac_recon[_hw_r][_hw_i, :, 1] + hw_mean[0, :, 1]

    _fig_hw2d = plt.figure(figsize=(5, 5))
    plt.plot(_orig_x, _orig_y, "b-", lw=2, label="original",alpha=0.3)
    plt.plot(_rec_x, _rec_y, "k-", lw=2, label=f"PARAFAC R={_hw_r}")
    plt.axis("equal")
    plt.legend()
    plt.title(
        f"Digit {hw_d_sl.value}, subject {hw_s_sl.value} — "
        f"R={_hw_r}, var explained {(1 - _hw_err**2) * 100:.1f}%"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
