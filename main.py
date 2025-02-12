import jax.numpy as jnp
from jax import jit, device_put
from src.config import GRID_SIZE, TIME_STEPS, PLANCK_CONSTANT, MASS, DIFFUSION_COEFFICIENT, DT, ENABLE_MIXED_PRECISION, USE_GPU
from src.quantum_potential import quantum_potential
from src.boundary_conditions import apply_periodic_boundary, apply_absorbing_boundary, apply_mixed_boundary
from src.eeg_processing import compute_power_spectrum
import logging

import jax

if USE_GPU:
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_xla_backend", "cuda")

# Enable mixed precision
if ENABLE_MIXED_PRECISION:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_dtype_bits", 16)


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dtype = jnp.float16 if ENABLE_MIXED_PRECISION else jnp.float32


@jit
def crank_nicholson_step(psi, V, Q, D):
    """Performs a single Crank-Nicholson step with GPU acceleration."""
    laplacian = D * (jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0) +
                     jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1) +
                     jnp.roll(psi, 1, axis=2) + jnp.roll(psi, -1, axis=2) - 6 * psi)
    psi = psi.astype(dtype) + DT * (
                - (PLANCK_CONSTANT / MASS) * laplacian.astype(dtype) + V.astype(dtype) * psi.astype(dtype) + Q.astype(
            dtype) * psi.astype(dtype))

    # Apply boundary conditions
    psi = apply_periodic_boundary(psi)
    psi = apply_absorbing_boundary(psi)
    psi = apply_mixed_boundary(psi)

    return psi


def run_simulation():
    """Runs the full EEG quantum wavefunction simulation."""
    logging.info("Initializing wavefunction...")
    psi = jnp.ones(GRID_SIZE, dtype=dtype) / jnp.sqrt(jnp.prod(jnp.array(GRID_SIZE, dtype=dtype)))

    for step in range(TIME_STEPS):
        logging.info(f"Running time step {step + 1}/{TIME_STEPS}")
        Q = quantum_potential(psi, DIFFUSION_COEFFICIENT).astype(dtype)
        V = jnp.zeros(GRID_SIZE, dtype=dtype)  # No external stimulus
        psi = crank_nicholson_step(psi, V, Q, DIFFUSION_COEFFICIENT)

    logging.info("Simulation complete. Computing EEG power spectrum...")
    freqs, psd = compute_power_spectrum(psi)

    return freqs, psd


if __name__ == "__main__":
    logging.info("Starting EEG quantum simulation...")
    freqs, psd = run_simulation()
    logging.info("Simulation finished. Power spectrum computed.")
