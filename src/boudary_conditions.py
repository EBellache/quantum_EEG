from config import *


def apply_periodic_bc(psi):
    """Apply periodic boundary conditions in (x, y) dimensions."""
    logging.debug("Applying periodic boundary conditions.")
    psi = psi.at[0, :, :].set(psi[-2, :, :])  # x-dimension
    psi = psi.at[-1, :, :].set(psi[1, :, :])
    psi = psi.at[:, 0, :].set(psi[:, -2, :])  # y-dimension
    psi = psi.at[:, -1, :].set(psi[:, 1, :])
    return psi


def apply_reflective_bc(psi):
    """Apply reflective boundary conditions in r (firing rate space)."""
    logging.debug("Applying reflective boundary conditions.")
    psi = psi.at[:, :, 0].set(psi[:, :, 1])  # Reflective at r=0
    psi = psi.at[:, :, -1].set(psi[:, :, -2])  # Reflective at r=R
    return psi
