from scipy.signal import welch
import jax.numpy as jnp
from config import EEG_SAMPLING_RATE, FFT_SEGMENTS


def compute_power_spectrum(psi, sampling_rate=EEG_SAMPLING_RATE, segments=FFT_SEGMENTS):
    """Compute EEG power spectral density, ensuring correct FFT segment size."""
    eeg_signal = jnp.abs(psi) ** 2
    freqs, psd = welch(eeg_signal.flatten(), sampling_rate, nperseg=segments)
    return freqs, psd
