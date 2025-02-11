from src.config import *
from src.solver import laplacian_3d


def quantum_potential(psi, D):
    """Computes the Macroscopic Quantum Potential (MQP) in 3D."""
    logging.debug("Computing Macroscopic Quantum Potential.")
    P = jnp.abs(psi) ** 2
    sqrt_P = jnp.sqrt(P)
    return (
        -(2 * D**2 / 1.0) * laplacian_3d(sqrt_P) / (sqrt_P + 1e-8)
    )  # Avoid division by zero
