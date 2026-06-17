# Lie Groups & Lie Algebras — Illustrated Notebooks

A set of self-contained notebooks that build intuition for Lie groups and Lie algebras by
*computing* with them: every adjoint, bracket, and exponential is calculated two ways (from the
definition and from a closed form) and cross-checked with `np.allclose`, so each notebook proves
its own math when you run it. All use the same dark-theme style and color conventions.

Each `*.py` is the source of truth (jupytext "percent" format); the paired `.ipynb` is generated
with `uv run jupytext --sync <file>.py`. See the repo `CLAUDE.md` for the workflow.

## Read in this order

The notebooks form a deliberate ladder — each adds one new wrinkle to the previous.

| # | Notebook | Group | What's new |
|---|----------|-------|------------|
| 1 | [`SO2_Lie_Theory`](SO2_Lie_Theory.ipynb) | **SO(2)** — planar rotations | The *abelian baseline*: 1-dimensional, bracket ≡ 0, both adjoints trivial. Ends with the isomorphism **SO(2) ≅ U(1)** (rotations as unit complex numbers). |
| 2 | [`SE2_Lie_Theory`](SE2_Lie_Theory.ipynb) | **SE(2)** — planar rigid motions | First *non-abelian* group: rotation + translation. Non-trivial bracket `[e₁,e₂]=e₃`, frame-dependent `Ad`, and the relation `Ad_exp(X) = exp(ad_X)`. |
| 3 | [`SO3_Lie_Theory`](SO3_Lie_Theory.ipynb) | **SO(3)** — 3D rotations | The rich case: the hat map turns the **cross product** into the bracket, Rodrigues' formula, the coincidences `ad_ŵ = ŵ` and `Ad_R = R`, and the **SU(2) double cover** (unit quaternions, 4π spinor periodicity). |
| 4 | [`SE3_Lie_Theory`](SE3_Lie_Theory.ipynb) | **SE(3)** — 3D rigid motions | Full 6-DOF: **twists**, **screw motions**, the left-Jacobian `V`, and 6×6 block-triangular adjoints — the operators robotics uses to move twists between frames. |
| 5 | [`SIM2_Lie_Theory`](SIM2_Lie_Theory.ipynb) | **SIM(2)** = ℂˣ ⋉ ℂ — planar similarities | The complex-number capstone: similarities are `z ↦ az + b`, and the *entire* Lie theory (exp, bracket, Ad) collapses into complex arithmetic. Signature geometry: the **logarithmic spiral**. |
| 6 | [`VectorField_View_Lie_Theory`](VectorField_View_Lie_Theory.ipynb) | (cross-cutting) | The unifying geometric lens: an algebra element *is a vector field*, `exp` *is a flow*, and the **bracket is the commutator of flows**. Connects three views of `𝔤` and the matrix picture to differential geometry. |
| 7 | [`Lorentz_Lie_Theory`](Lorentz_Lie_Theory.ipynb) | **SO⁺(1,3)** — spacetime symmetry | The physics capstone: boosts are *hyperbolic* rotations (split-complex numbers, rapidity, `v=tanh φ`); `[K,K]~J` gives the **Thomas–Wigner rotation**; boost orbits are Rindler hyperbolas; and `SL(2,ℂ)` double-covers it (spinors, the Dirac equation). Ties the complex, bracket, and vector-field threads together. |

## Two threads running through the set

- **The "complex numbers" thread.** Rotation groups are secretly about complex/quaternionic numbers:
  **SO(2) ≅ U(1)** (unit circle, §1 of nb 1) → **ℂˣ** and **SU(2)** (all scale-rotations / the 2:1
  cover, nb 3, 5) → **SIM(2) = ℂˣ ⋉ ℂ** (nb 5). SE(2) is the `|a|=1` subgroup of SIM(2); SO(2) is the
  further `b=0` subgroup. The Lorentz group (nb 7) extends the thread to **split-complex numbers**
  (`j²=+1`): a boost is "multiplication by `e^{jφ}`", the hyperbolic mirror of `e^{iθ}`, and `SL(2,ℂ)`
  is the relativistic version of the `SU(2)` cover.
- **The "non-commutativity / bracket" thread.** The bracket measures the failure of two motions to
  commute. It shows up as the *group commutator* — the gap between `exp(sX)exp(tY)` and
  `exp(tY)exp(sX)` (nb 2 §6, nb 3 §6, nb 4 §6, and nb 7's **Thomas–Wigner rotation** `[K,K]~J`) — and
  as the *flow commutator* — the non-closing vector-field quadrilateral (nb 6 §2). These are two
  faces of the same coin.

## Reference papers

Background reading for the math (PDFs in this directory):

- **`3DGeometry-notes.pdf`** — Drummond's notes. *Read these first.*
- **`eade-lie.pdf`** (text in `eade-lie.md`), **`eade-exp-diff.pdf`**, **`eade_lie_groups_computer_vision.pdf`** —
  Ethan Eade's notes on Lie groups for 2D/3D transformations. Useful, but contain some errors —
  e.g. they don't always distinguish the group adjoint `Ad` (on the group) from the algebra adjoint
  `ad` (on the algebra). Cross-check against the notebooks above, which keep the two separate.
- **`lie-grous-roboticists-arxiv1812.01537v9.pdf`** — "A micro Lie theory for state estimation in
  robotics" (Solà et al.) — a clean modern reference for the SE(3) conventions used in notebook 4.
