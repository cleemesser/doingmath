# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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

# %%
# @MISC {240492,
#     TITLE = {Relationship between SVD and PCA. How to use SVD to perform PCA?},
#     AUTHOR = {turdus-merula (https://stats.stackexchange.com/users/92560/turdus-merula)},
#     HOWPUBLISHED = {Cross Validated},
#     NOTE = {URL:https://stats.stackexchange.com/q/240492 (version: 2022-10-09)},
#     EPRINT = {https://stats.stackexchange.com/q/240492},
#     URL = {https://stats.stackexchange.com/q/240492}
# }
# %%
import numpy as np
from numpy import linalg as la

np.random.seed(42)
# %% visualization libraries
import altair as alt
import pandas as pd
import wigglystuff
# %%
import clmmathtools.viz as mv
from clmmathtools.viz import palette

mv.set_interactive("auto")

# %% helper functions


def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs


# %%

# Let the data matrix X be of n x p size,
# where n is the number of samples and p is the number of variables
n, p = 5, 3
X = np.random.rand(n, p)
# Let us assume that it is centered
X -= np.mean(X, axis=0)
# %%

# %%

# %%
# the p x p covariance matrix
C = np.cov(X, rowvar=False)
print("C = \n", C)
# C is a symmetric matrix and so it can be diagonalized:
l, principal_axes = la.eig(C)
# sort results wrt. eigenvalues
idx = l.argsort()[::-1]
l, principal_axes = l[idx], principal_axes[:, idx]
# the eigenvalues in decreasing order
print("l = \n", l)
# a matrix of eigenvectors (each column is an eigenvector)
print("V = \n", principal_axes)
# projections of X on the principal axes are called principal components
principal_components = X.dot(principal_axes)
print("Y = \n", principal_components)

# we now perform singular value decomposition of X
# "economy size" (or "thin") SVD
U, s, Vt = la.svd(X, full_matrices=False)
V = Vt.T
S = np.diag(s)

# 1) then columns of V are principal directions/axes.
assert np.allclose(*flip_signs(V, principal_axes))

# 2) columns of US are principal components
assert np.allclose(*flip_signs(U.dot(S), principal_components))

# 3) singular values are related to the eigenvalues of covariance matrix
assert np.allclose((s**2) / (n - 1), l)

# 8) dimensionality reduction
k = 2
PC_k = principal_components[:, 0:k]
US_k = U[:, 0:k].dot(S[0:k, 0:k])
assert np.allclose(*flip_signs(PC_k, US_k))

# 10) we used "economy size" (or "thin") SVD
assert U.shape == (n, p)
assert S.shape == (p, p)
assert V.shape == (p, p)
# %%
# %% [markdown]
# ## Seeing it in 3D
#
# `X` is 5 points in R^3, so the whole story of this notebook fits in one picture: the
# points, the best-fit plane spanned by the first two principal axes, the residuals
# dropped onto that plane, and the three principal axes scaled by their spread.
#
# `mv.set_interactive("auto")` gives a live, orbitable widget in a notebook and falls back
# to a static PNG when run headless (so `jupytext --execute` still works). Force one or the
# other with `sc.display(interactive=True)` / `sc.display(interactive=False)`.
# %%
# spread along each principal axis: s_i / sqrt(n-1) is the std dev of the i-th PC
sd = s / np.sqrt(n - 1)

# projector onto the best-fit (PC1, PC2) plane, and the feet of the residuals
P2 = V[:, 0:k] @ V[:, 0:k].T
X_proj = X @ P2.T

# How big to draw the plane. Sizing by standard deviations is statistically meaningful but
# the multiples are arbitrary; clipping to the convex hull of X_proj would be truer to the
# data but ragged with only n = 5 points. Tweak to taste.
half_u, half_v = 1.3 * sd[0], 1.9 * sd[1]

sc = mv.Space3D(bounds=1.1, elev=28, azim=-118)

# the best-fit plane, centered on the (already centered) data
sc.parallelogram(
    -half_u * V[:, 0] - half_v * V[:, 1],
    2 * half_u * V[:, 0],
    2 * half_v * V[:, 1],
    facecolor=palette.BLUE,
    alpha=0.20,
    edgecolor=palette.GREY,
)

# residual of each point: the part of x living along the discarded axis PC3
for x, x_proj in zip(X, X_proj):
    sc.line([x_proj, x], color=palette.GREY, width=1.5, alpha=0.8)

sc.points(X, color=palette.ORANGE, size=14)

# principal axes, each drawn 2 std devs long: PC1 red, PC2 green (both in the plane),
# PC3 yellow (the plane's normal -- the direction PCA throws away at k = 2)
for i, color in enumerate((palette.RED, palette.GREEN, palette.YELLOW)):
    sc.arrow([0, 0, 0], 2 * sd[i] * V[:, i], color=color, width=1.2, alpha=0.2)

if mv.in_notebook:
    from IPython.display import display

    display(sc.display())
else:
    sc.display(interactive=True, vedo_display=True)

# %%
# Eckart-Young: the error of the best rank-k approximation is exactly the norm of the
# singular values you dropped. Here k = 2, so that is just s[2] -- the length of the
# yellow arrow's worth of variance, and the total length of the grey residual stubs.
assert np.allclose(la.norm(X - X_proj), la.norm(s[k:]))
print("residual norms =", np.round(la.norm(X - X_proj, axis=1), 4))
print(
    "||X - X_k||_F =",
    round(float(la.norm(X - X_proj)), 4),
    "= s[2] =",
    round(float(s[2]), 4),
)

# %%
# blobs database
import sklearn

Xb, yb = sklearn.datasets.make_blobs(
    centers=2, n_features=3, cluster_std=2.0, random_state=42
)
Xb -= np.mean(Xb, axis=0)


# %%

Ub, sb, Vhb = la.svd(Xb, full_matrices=False)
nb = len(Xb)

# %%
# spread along each principal axis: s_i / sqrt(n-1) is the std dev of the i-th PC
sd_b = sb / np.sqrt(nb - 1)

# %%
# projector onto the best-fit (PC1, PC2) plane, and the feet of the residuals
P2b = Vhb.T[:, 0:k] @ Vhb.T[:, 0:k].T
Xb_proj = Xb @ P2b.T


# %%

# How big to draw the plane. Sizing by standard deviations is statistically meaningful but
# the multiples are arbitrary; clipping to the convex hull of X_proj would be truer to the
# data but ragged with only n = 5 points. Tweak to taste.
half_u, half_v = 1.3 * sd_b[0], 1.9 * sd_b[1]


# %%

sc = mv.Space3D(bounds=1.1, elev=28, azim=-118)

# the best-fit plane, centered on the (already centered) data
sc.parallelogram(
    -half_u * Vhb.T[:, 0] - half_v * Vhb.T[:, 1],
    2 * half_u * Vhb.T[:, 0],
    2 * half_v * Vhb.T[:, 1],
    facecolor=palette.BLUE,
    alpha=0.20,
    edgecolor=palette.GREY,
)

# residual of each point: the part of x living along the discarded axis PC3
for x, x_proj in zip(Xb, Xb_proj):
    sc.line([x_proj, x], color=palette.GREY, width=1.5, alpha=0.8)

sc.points(Xb, color=palette.ORANGE, size=4)

# principal axes, each drawn 2 std devs long: PC1 red, PC2 green (both in the plane),
# PC3 yellow (the plane's normal -- the direction PCA throws away at k = 2)
for i, color in enumerate((palette.RED, palette.GREEN, palette.YELLOW)):
    sc.arrow([0, 0, 0], 2 * sd_b[i] * Vhb.T[:, i], color=color, width=1.2, alpha=0.2)

# sc.display(interactive=True, vedo_display=True)
if mv.in_notebook():
    from IPython.display import display

    display(sc.display(interactive=True))
else:
    sc.display(interactive=True, vedo_display=True)

# %%
