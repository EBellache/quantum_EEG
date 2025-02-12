# src/neural_wavefunction.py
import jax.numpy as jnp
from jax import jit
from config import GRID_SIZE, TIME_STEPS, PLANCK_CONSTANT, MASS, DIFFUSION_COEFFICIENT, DT, ENABLE_MIXED_PRECISION
from quantum_potential import quantum_potential
from boundary_conditions import apply_periodic_boundary, apply_absorbing_boundary, apply_mixed_boundary

dtype = jnp.float16 if ENABLE_MIXED_PRECISION else jnp.float32


@jit
def crank_nicholson_step(psi, V, Q, D):
    """Performs a single Crank-Nicholson step with hybrid boundary conditions."""
    laplacian = D * (jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0) +
                     jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1) +
                     jnp.roll(psi, 1, axis=2) + jnp.roll(psi, -1, axis=2) - 6 * psi)

    psi = psi + DT * (- (PLANCK_CONSTANT / MASS) * laplacian + V * psi + Q * psi)

    # Apply hybrid boundary conditions
    psi = apply_periodic_boundary(psi)  # Periodic in x
    psi = apply_absorbing_boundary(psi)  # Absorbing in y, r
    psi = apply_mixed_boundary(psi)  # Edge stabilization

    return psi


def simulate_neural_wavefunction():
    """Runs the EEG wavefunction simulation with optimized settings."""
    psi = jnp.ones(GRID_SIZE, dtype=dtype) / jnp.sqrt(jnp.prod(jnp.array(GRID_SIZE, dtype=dtype)))

    for _ in range(TIME_STEPS):
        Q = quantum_potential(psi, DIFFUSION_COEFFICIENT).astype(dtype)
        V = jnp.zeros(GRID_SIZE, dtype=dtype)  # No external stimulus
        psi = crank_nicholson_step(psi, V, Q, DIFFUSION_COEFFICIENT)

    return psi
