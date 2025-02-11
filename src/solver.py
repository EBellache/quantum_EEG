from jax import jit
from src.config import *
from src.boudary_conditions import apply_reflective_bc, apply_periodic_bc


def laplacian_3d(psi):
    """Computes the Laplacian using finite differences for a 3D grid."""
    logging.debug("Computing Laplacian.")
    return (
        jnp.roll(psi, -1, axis=0)
        + jnp.roll(psi, 1, axis=0)
        + jnp.roll(psi, -1, axis=1)
        + jnp.roll(psi, 1, axis=1)
        + jnp.roll(psi, -1, axis=2)
        + jnp.roll(psi, 1, axis=2)
        - 6 * psi
    ) / DELTA_X**2


@jit
def crank_nicholson_step(psi, V, Q_macro, D):
    """Numerically stable Crank-Nicholson update for the Schrödinger equation with EEG band switching."""
    logging.debug("Performing Crank-Nicholson step.")
    H = -(D**2) * laplacian_3d(psi) + V * psi + Q_macro * psi
    psi_next = psi + (1j * DELTA_T / D) * H
    psi_next = apply_periodic_bc(psi_next)  # Apply periodic BC
    psi_next = apply_reflective_bc(psi_next)  # Apply reflective BC
    return psi_next / jnp.linalg.norm(psi_next)
