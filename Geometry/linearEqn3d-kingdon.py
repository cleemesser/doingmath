# ---
# jupyter:
#   jupytext:
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
from kingdon import Algebra
import numpy as np
import sympy as sp

sp.init_printing()
# should probably just import these from somewhere
import matplotlib.colors as mcolors

GREY = 0x888888
RED = 0xFF3333
GREEN = 0x33FF33
BLUE = 0x3388FF
ORANGE = 0xFF9933
YELLOW = 0xFFDD33
BLACK = 0x000000
PURPLE = mcolors.CSS4_COLORS["purple"]


pga3 = Algebra(3, 0, 1)
locals().update(pga3.blades)  # blades e0, e1..
pga3.blades

# %%
A_list = [[1, 1, -2], [2, -3, 1], [4, 1, -1]]
A = np.array(A_list)

# A = np.eye(3) # this is the orthogonalized A our goal in full row-eschelon manipulations
# A[0,0] = 2
sp.Matrix(A)

# %%

plane = lambda a, b, c, d: a * e1 + b * e2 + c * e3 + d * e0  # a x + b y + c z + d = 0
plane0 = lambda a, b, c: a * e1 + b * e2 + c * e3  # a x + b y + c z = 0; d = 0

# %%
plane0(*A_list[0])

# %%
# this only runs in the notebook
pga3.graph(
    BLUE,
    plane0(*A[0, :]),
    "row 0 plane",
    YELLOW,
    plane0(*A[1, :]),
    "row 1 plane",
    GREEN,
    plane0(*A[2, :]),
    "row 2 plane",
    RED,
    plane0(*A[0, :]) ^ plane0(*A[1, :]),
    "line 01",
    RED,
    plane0(*A[1, :]) ^ plane0(*A[2, :]),
    "line 12",
    RED,
    plane0(*A[0, :]) ^ plane0(*A[2, :]),
    "line 02",
    BLACK,
    plane0(*A[0, :]) ^ plane0(*A[1, :]) ^ plane0(*A[2, :]),
    "intersection",
    grid=1,
)

# %%

P1 = plane(1, 0, 0, -1)  # x = 1
P2 = plane(0, 1, 0, -2)  # y = 2
P3 = plane(1, 1, 1, -9)  # x + y + z = 9

L12, L13, L23 = P1 ^ P2, P1 ^ P3, P2 ^ P3  # three pairwise lines (grade 2)
Pt = P1 ^ P2 ^ P3  # common point (grade 3)

# recover Euclidean coords: with weight w = coeff(e123),
# (x,y,z) = (-e023/w, +e013/w, -e012/w)
c = {pga3.bin2canon[k]: v for k, v in zip(Pt.keys(), Pt.values())}
w = c["e123"]
xyz = np.array([-c["e023"] / w, c["e013"] / w, -c["e012"] / w])
print(xyz)  # -> [1. 2. 6.]

# %%
