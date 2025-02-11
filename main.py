from src.config import *
from src.neural_wavefunction import run_simulation

if __name__ == "__main__":
    logging.info("Starting Quantum RSNN EEG Simulation.")
    run_simulation()
    logging.info("Simulation completed successfully.")
