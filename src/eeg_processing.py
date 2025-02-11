from scipy.fft import fftn, fftshift
import jax.numpy as jnp


def compute_eeg_spectrum(psi):
    """Computes the EEG frequency spectrum using 3D FFT."""
    spectrum = jnp.abs(fftshift(fftn(jnp.abs(psi) ** 2)))  # Power spectrum
    return jnp.log1p(spectrum)  # Log transform for visualization
