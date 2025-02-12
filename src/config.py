import jax.numpy as jnp

# Simulation Grid (Optimized)
GRID_SIZE = (256, 256, 128)  # High-resolution cortical modeling
TIME_STEPS = 20000  # Long-term neural coherence tracking
DT = 0.0005  # Time step for stability

# Schrödinger Equation Parameters
PLANCK_CONSTANT = 1.0
MASS = 1.0
DIFFUSION_COEFFICIENT = 0.3

# Macroscopic Quantum Potential (MQP)
MQP_SCALING = 2.0
MQP_SLEEP_FACTOR = 6.0

# Synaptic Connectivity (Log-Normal Distribution)
CONNECTIVITY_MEAN = 0.0
CONNECTIVITY_STD = 1.2
SPARSE_THRESHOLD = 0.005

# EEG Processing
EEG_SAMPLING_RATE = 512
FFT_SEGMENTS = 512

# Compute Hardware
USE_GPU = True
NUM_CPU_THREADS = 64
ENABLE_MIXED_PRECISION = True
