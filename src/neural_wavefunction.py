from src.config import *
from src.quantum_potential import quantum_potential
from src.solver import crank_nicholson_step
from src.frequency_dynamics import get_dynamic_diffusion


def run_simulation():
    """Runs the 3D quantum RSNN simulation with EEG band transitions."""
    logging.info("Starting neural wavefunction simulation.")
    psi = jnp.ones(GRID_SIZE, dtype=jnp.complex64) / jnp.sqrt(
        GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]
    )

    for time_step in range(TIME_STEPS):
        D = (
            EEG_BANDS["alpha"]
            if not DYNAMICS_MODE
            else get_dynamic_diffusion(time_step)
        )
        Q_macro = quantum_potential(psi, D)
        V = jnp.zeros(GRID_SIZE)  # External potential (can be dynamic)
        psi = crank_nicholson_step(psi, V, Q_macro, D)
        if time_step % 100 == 0:
            logging.info(f"Completed time step {time_step}/{TIME_STEPS}")
    return psi
