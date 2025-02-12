import numpy as np
import jax.numpy as jnp
from config import GRID_SIZE, CONNECTIVITY_MEAN, CONNECTIVITY_STD, SPARSE_THRESHOLD


def generate_log_normal_connectivity(grid_size=GRID_SIZE):
    """Creates a log-normal synaptic weight matrix, ensuring it matches the grid size."""
    num_neurons = jnp.prod(jnp.array(grid_size))
    weights = np.random.lognormal(CONNECTIVITY_MEAN, CONNECTIVITY_STD, (num_neurons, num_neurons))
    weights[weights < SPARSE_THRESHOLD] = 0  # Enforce sparsity
    return jnp.array(weights)
