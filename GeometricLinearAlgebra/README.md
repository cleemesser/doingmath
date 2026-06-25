# Geometric Linear Algebra — A Coordinate-Free, Illustrated Introduction

A series of self-contained notebooks that build linear algebra **geometrically**, from arrows in the
plane up to operators in space — and, as far as possible, **without coordinates**. The guiding idea:
every important object (projection, rotation, the dot and wedge products, area, volume, the
determinant, the trace) has a *geometric* definition that comes first; **coordinates, bases, and
matrices are introduced only at the end**, as one convenient way to write those operators down. Along
the way the complex numbers fall out on their own — as exactly the linear maps that are a scaling
combined with a pure rotation.

Like the sibling [`LieGroups/`](../LieGroups/README.md) set, each notebook *computes* with what it
describes and **cross-checks its own claims with `np.allclose`**, so each notebook proves its own math
when you run it. Visualization is **interactive** via [`k3d`](https://k3d-jupyter.org/) (WebGL): plane
geometry is drawn in the `z = 0` plane and viewed top-down, so the very same tools carry over
seamlessly when we move into 3D.

Each `*.py` is the source of truth (jupytext "percent" format); the paired `.ipynb` is generated with
`uv run jupytext --sync <file>.py`. See the workflow note at the bottom.

## Read in this order

The notebooks form a deliberate ladder — coordinate-free geometry first, coordinates last.

| # | Notebook | What's new |
|---|----------|------------|
| 1 | [`01_Linearity_Lines_to_Lines`](01_Linearity_Lines_to_Lines.py) | The foundation: a vector is an *arrow* you can **add** and **scale**; a map is **linear** iff it respects both. The headline theorem — **lines go to lines**, parallels stay parallel, the origin is fixed — proved from the definition and watched live on a deforming grid. 1D linear maps are pure scalings; affine ≠ linear. |
| 2 | `02_Coordinate_Free_Operators` *(planned)* | Define the canonical operators **geometrically, without a basis**: **projection** onto a vector, **rotation**, **scaling**, and **shear**. Each built from lengths and the right-angle relation; verified linear and composed. |
| 3 | `03_Dot_and_Wedge` *(planned)* | Two bilinear measures of a pair of arrows: the **dot product** (length & angle, the symmetric part) and the **wedge product** (signed area, the antisymmetric part). Lengths, angles, areas — all coordinate-free. |
| 4 | `04_Volume_Determinant_Trace` *(planned)* | The **volume element** as the top-degree wedge; the **determinant** defined as the factor by which an operator scales volume (signed); and a **coordinate-free trace** as the rate of volume change / the first-order stretch. 2D and 3D. |
| 5 | `05_Bases_Coordinates_Complex` *(planned)* | *Now* introduce **basis vectors**, **coordinates** of a vector, and the **matrix** of an operator. Recover determinant and trace as the familiar formulas. Capstone: **discover ℂ** — the maps that are "scale × pure rotation" form a plane closed under composition, and that algebra *is* the complex numbers (a twist). |

## The thread running through the set

- **Coordinate-free first.** Operators and measures are defined by what they *do* to arrows (project,
  rotate, measure area/volume), not by arrays of numbers. The determinant is "how much volume scales,"
  the trace is "first-order volume rate," long before any matrix appears.
- **The complex-number payoff.** The capstone mirrors the `LieGroups/` "complex numbers" thread from
  the other direction: there, **SO(2) ≅ U(1)** emerges from rotation *groups*; here, **ℂ** emerges from
  the *linear operators* that combine a scaling with a pure rotation — a scale-and-twist. Same secret,
  approached from elementary geometry.

## Workflow

```bash
uv sync                                                   # set up the environment (Python ≥ 3.14)
uv run jupytext --sync GeometricLinearAlgebra/01_*.py     # regenerate the paired .ipynb
uv run jupyter lab                                        # open and run interactively (k3d needs a live frontend)
```

k3d figures are interactive widgets and only render in a live Jupyter frontend (Lab/Notebook). Drag to
orbit, scroll to zoom.
