from jax import jit
from config import *


@jit
def quantum_potential(psi, D=DIFFUSION_COEFFICIENT):
    """Computes MQP with log-normal connectivity adjustments, ensuring D is used."""
    P = jnp.abs(psi) ** 2
    sqrt_P = jnp.sqrt(P)
    laplacian = D * jnp.nan_to_num(jnp.gradient(sqrt_P), nan=0)
    return - (2 * D**2) * laplacian / (sqrt_P + 1e-8)