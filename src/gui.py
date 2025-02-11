import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neural_wavefunction import simulate_neural_wavefunction
from eeg_processing import compute_eeg_spectrum, plot_power_spectrum
import logging


def run_simulation():
    """Runs the Quantum RSNN EEG simulation and updates GUI with emergent frequency spectrum."""
    logging.info("Running simulation without predefined EEG bands.")
    psi = simulate_neural_wavefunction()
    dt = 0.001  # Time step
    freqs, spectrum = compute_eeg_spectrum(psi, dt)
    update_plot(freqs, spectrum)


def update_plot(freqs, spectrum):
    """Update the GUI plot with the computed EEG frequency spectrum."""
    ax.clear()
    ax.plot(freqs, spectrum, label="EEG Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Emergent EEG Frequency Quantization")
    ax.legend()
    canvas.draw()


# Initialize Tkinter GUI
root = tk.Tk()
root.title("Quantum RSNN 3D EEG Simulation (Emergent Frequencies)")

fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

btn_run = tk.Button(root, text="Run Simulation", command=run_simulation)
btn_run.pack()

root.mainloop()
