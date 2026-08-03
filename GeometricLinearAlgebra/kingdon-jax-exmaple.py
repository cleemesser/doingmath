import jax
import jax.numpy as jnp
import kingdon

# 1. Initialize a standard 2D Clifford Algebra space (p=2, q=0)
alg = kingdon.Algebra(2, 0)


# 2. Define a pure function processing geometric algebra operations
def geometric_loss(coeff_a, coeff_b):
    # Construct Multivectors inside the function using JAX arrays
    mv_a = alg.multivector(e1=coeff_a[0], e2=coeff_a[1])
    mv_b = alg.multivector(e1=coeff_b[0], e2=coeff_b[1])

    # Perform geometric product
    product = mv_a * mv_b

    # Return a scalar value (e.g., the pseudoscalar component 'e12')
    return product["e12"]


# 3. Setup sample data arrays
a = jnp.array([1.5, 2.5])
b = jnp.array([3.0, -1.0])

# 4. Apply JAX transformations: JIT-compile and compute gradients
grad_fn = jax.jit(jax.grad(geometric_loss, argnums=(0, 1)))
grads_a, grads_b = grad_fn(a, b)

print("Gradients with respect to a:", grads_a)
print("Gradients with respect to b:", grads_b)
