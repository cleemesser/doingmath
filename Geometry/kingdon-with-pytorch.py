# ---
# jupyter:
#   jupytext:
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
import sys
import torch

sys.platform

# %%

# 1. Print version
print(f"PyTorch version: {torch.__version__}")

# 2. Check for Apple Silicon GPU Acceleration
if sys.platform == "darwin":
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print("Success! MPS (GPU) is available. Matrix:", x)
    else:
        print("MPS device not found. Running on CPU.")

# %%
import torch
from kingdon import Algebra

# Create a 3D Euclidean Geometric Algebra (3, 0, 0)
alg = Algebra(3)

# Define vector coefficients as PyTorch tensors with gradient tracking
coeffs_v1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
coeffs_v2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Instantiate Kingdon multivectors by mapping tensors to specific basis blades
# e1, e2, e3 represent the standard basis vectors
v1 = alg.multivector(e1=coeffs_v1[0], e2=coeffs_v1[1], e3=coeffs_v1[2])
v2 = alg.multivector(e1=coeffs_v2[0], e2=coeffs_v2[1], e3=coeffs_v2[2])

# %%
# Geometric Product
gp_result = v1 * v2

# Wedge (Outer) Product -> Results in a Bivector (e12, e13, e23)
wedge_result = v1 ^ v2

# Inner (Dot) Product -> Results in a Scalar
dot_result = v1 | v2

print("Geometric Product:", gp_result)
print("Outer Product blades:", wedge_result.keys())

# %%
# gp_result.norm()
type(gp_result)

# %%
N = 5
xvals = torch.rand(3, N)
xvals

# %%
# N = 5
# xvals = np.random.rand(3, N)
x = alg.vector(xvals)
x

# %%
x.shape

# %%
len(x)

# %%
d = x.norm()
d

# %%
x[d.e > 1]

# %%
pga3d = Algebra(3, 0, 1)
v = pga3d.vector(vertices.T).dual()
v.shape

# %%
facets = v[faces]
facets.shape

# %%
planes = facets[..., 0] & facets[..., 1] & facets[..., 2]
planes.shape
# (4, M)

# %%
areas = planes.norm()
area = areas.map(np.sum)

# %%
volume = np.sum(planes.e0)

# %%
