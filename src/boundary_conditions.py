import jax.numpy as jnp


def apply_periodic_boundary(psi):
    """Applies periodic boundary conditions in the x-direction."""
    psi = jnp.pad(psi, pad_width=((1, 1), (0, 0), (0, 0)), mode='wrap')
    return psi[1:-1, :, :]


def apply_absorbing_boundary(psi, decay_factor=0.95):
    """Applies absorbing boundary conditions along y and r axes."""
    y, r = psi.shape[1], psi.shape[2]
    absorption_mask = jnp.exp(
        -decay_factor * (jnp.linspace(0, 1, y)[:, None] ** 2 + jnp.linspace(0, 1, r)[None, :] ** 2))
    return psi * absorption_mask


def apply_mixed_boundary(psi, alpha=0.1):
    """Applies a mixed absorbing-reflecting boundary condition."""
    boundary_mask = jnp.exp(-alpha * jnp.abs(psi))
    return psi * boundary_mask
