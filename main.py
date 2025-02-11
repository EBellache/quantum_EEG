from src.config import *
from src.neural_wavefunction import simulate_neural_wavefunction
from src.eeg_processing import compute_eeg_spectrum, plot_power_spectrum

if __name__ == "__main__":
    logging.info("Starting Quantum RSNN EEG Simulation (no predefined EEG bands).")

    psi = simulate_neural_wavefunction()  # Run simulation

    # Extract and visualize emergent frequencies
    dt = 0.001  # Time step
    freqs, spectrum = compute_eeg_spectrum(psi, dt)
    plot_power_spectrum(freqs, spectrum)

    logging.info("Simulation completed successfully.")
