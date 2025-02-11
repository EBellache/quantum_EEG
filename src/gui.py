import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.neural_wavefunction import simulate_neural_wavefunction
from src.eeg_processing import compute_eeg_spectrum


def run_simulation():
    psi = simulate_neural_wavefunction()
    spectrum = compute_eeg_spectrum(psi)
    update_plot(spectrum)


def update_plot(spectrum):
    ax.clear()
    ax.imshow(spectrum[:, :, 32], cmap="inferno", aspect="auto")
    ax.set_title("EEG Power Spectrum (Slice)")
    canvas.draw()


root = tk.Tk()
root.title("Quantum RSNN 3D EEG Simulation")
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
btn_run = tk.Button(root, text="Run Simulation", command=run_simulation)
btn_run.pack()
root.mainloop()
