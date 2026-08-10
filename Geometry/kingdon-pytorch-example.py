import torch
import kingdon

# 1. Initialize a 3D Projective Geometric Algebra (PGA) space
# Metric for 3D PGA: 3 positive, 0 negative, 1 zero dimensions (p=3, q=0, r=1)
alg = kingdon.Algebra(3, 0, 1)

# 2. Define coefficients as PyTorch tensors with gradient tracking enabled
# We define a point and a translator plane
coeff_point = torch.tensor(
    [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True
)
coeff_plane = torch.tensor(
    [0.0, 1.0, 0.0, -2.0], dtype=torch.float32, requires_grad=True
)

# 3. Instantiate MultiVectors using the PyTorch tensors
# We assign coefficients to specific basis blades of the algebra
point = alg.multivector(
    e123=coeff_point[0], e032=coeff_point[1], e013=coeff_point[2], e021=coeff_point[3]
)
plane = alg.multivector(
    e1=coeff_plane[0], e2=coeff_plane[1], e3=coeff_plane[2], e0=coeff_plane[3]
)

# 4. Perform a geometric product or sandwich operation (reflection)
reflected_point = plane * point * plane

# 5. Extract a scalar value to optimize (e.g., the e123 component)
loss = reflected_point["e123"]

# 6. Compute gradients using PyTorch Autograd
loss.backward()

# Access computed gradients directly from the original tensors
print("Point gradients:", coeff_point.grad)
print("Plane gradients:", coeff_plane.grad)
