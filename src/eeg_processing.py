from scipy.fft import fftn, fftshift
import jax.numpy as jnp

from scipy.fft import fftn, fftfreq
import matplotlib.pyplot as plt


def compute_eeg_spectrum(psi, dt):
    """Compute FFT-based EEG frequency spectrum."""
    spectrum = fftn(jnp.abs(psi) ** 2, axes=[-1])  # FFT along time
    freqs = fftfreq(psi.shape[-1], d=dt)
    return freqs, jnp.abs(spectrum)


def plot_power_spectrum(freqs, spectrum):
    """Plot the emergent EEG spectrum."""
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spectrum, label="EEG Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Emergent EEG Frequency Quantization")
    plt.legend()
    plt.show()
