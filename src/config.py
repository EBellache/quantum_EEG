import logging
import jax.numpy as jnp

# Define grid size and simulation parameters
GRID_SIZE = (64, 64, 64)  # 3D grid (x, y, r)
TIME_STEPS = 2000  # Number of time steps
DELTA_T = 0.001  # Time step size
DELTA_X = 0.1  # Spatial step size

D = 1.0  # Use a constant diffusion coefficient for emergent frequency modes

DYNAMICS_MODE = True  # Enable dynamic EEG band switching

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
