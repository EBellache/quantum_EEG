from src.config import *
from src.quantum_potential import quantum_potential
from src.solver import crank_nicholson_step


def simulate_neural_wavefunction():
    """Runs the 3D quantum RSNN simulation without predefined EEG bands."""
    logging.info("Starting neural wavefunction simulation (no predefined EEG bands).")
    psi = jnp.ones(GRID_SIZE, dtype=jnp.complex64) / jnp.sqrt(
        GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]
    )

    for time_step in range(TIME_STEPS):
        Q_macro = quantum_potential(psi, D)  # Compute MQP dynamically
        V = jnp.zeros(GRID_SIZE)  # No external forced potential
        psi = crank_nicholson_step(psi, V, Q_macro, D)

        if time_step % 100 == 0:
            logging.info(f"Completed time step {time_step}/{TIME_STEPS}")

    return psi
