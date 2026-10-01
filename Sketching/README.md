# Low-Rank Functions and Sketching

How a complicated, information-dense function of several variables can be represented by a short sum
of products of one-variable functions — a low-rank matrix, or a CP / Tucker / tensor-train tensor —
and how such representations are *found* cheaply, by random projection or by sampling a few rows and
columns, without ever computing on the full object.

Like the rest of the repository, the notebook computes what it describes and checks its own claims.
The `.py` file is the source of truth (jupytext percent format); generate the paired notebook with
`uv run jupytext --sync Sketching/Low_Rank_Functions_and_Sketching.py`.

## A note on terminology

"Sketching" covers two related things. The **representation** — a low-rank factorization or a tensor
format — is the compressed object. **Sketching** in the narrow numerical-linear-algebra sense is the
*method*: multiply the large object by a small random matrix and recover a low-rank approximation from
the product. The notebook takes them in that order: why functions are low rank, then how to find the
low-rank form.

## What it shows

| § | Topic | Measured result |
|---|---|---|
| 1 | Rank of a sampled function; Eckart–Young | $1/(1+x+y)$ needs 7 terms for 12 digits on a 400×400 grid; $\mathbf 1[x<y]$ is full rank while $\mathbf 1[x<0.5]$ is rank 1 — low rank is about alignment with the axes, not smoothness |
| 2 | Johnson–Lindenstrauss | 10,000-dimensional points projected to 400 dimensions keep all 19,900 distances within ~15%; distortion falls as $1/\sqrt k$, independent of the ambient dimension |
| 3 | Randomized SVD (Halko–Martinsson–Tropp) | optimal at once for fast spectral decay; power iteration fixes slow decay; power iteration *without* re-orthonormalization makes the error ~300× worse |
| 4 | Single-pass sketch (Tropp–Yurtsever–Udell–Cevher) | compresses from one pass and absorbs streaming updates, at 5–10× the optimal error |
| 5 | Adaptive cross approximation | the 400×400 function recovered to 1.5e-12 from 4% of its values; the explicit skeleton formula loses eight digits through an ill-conditioned inverse that ACA's elimination avoids |
| 6 | CP and Tucker | $\sin(x+y+z)$ has Tucker rank (2,2,2), complex CP rank 2, real CP rank 3 (certified by the hyperdeterminant); the $W$ tensor has no best rank-2 approximation, and ALS degenerates on it |
| 7 | Tensor trains | $\sin(\sum_i x_i)$ has TT rank 2 in every dimension; storage linear rather than exponential in $d$ |
| 8 | Applications | PARAFAC for EEG, CP/low-rank compression of networks (LoRA), spectra of latent representations |

## Where to start reading

- N. Halko, P.-G. Martinsson and J. A. Tropp, "Finding structure with randomness", *SIAM Review*
  **53** (2011) — the clearest entry point to randomized low-rank methods.
- T. G. Kolda and B. W. Bader, "Tensor decompositions and applications", *SIAM Review* **51**
  (2009) — the standard survey of CP and Tucker.
- V. de Silva and L.-H. Lim, "Tensor rank and the ill-posedness of the best low-rank approximation
  problem", *SIAM J. Matrix Anal. Appl.* **30** (2008) — why tensor rank behaves so differently from
  matrix rank.
- P.-G. Martinsson and J. A. Tropp, "Randomized numerical linear algebra", *Acta Numerica* **29**
  (2020) — the comprehensive modern treatment.

The notebook's own reference list gives the original sources for each section.

## Running

```bash
uv sync
MPLBACKEND=Agg uv run python Sketching/Low_Rank_Functions_and_Sketching.py   # headless check
uv run jupytext --sync --execute Sketching/Low_Rank_Functions_and_Sketching.py
```

Uses numpy, matplotlib and tensorly (for CP/Tucker), all base dependencies.
