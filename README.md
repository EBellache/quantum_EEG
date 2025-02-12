# Quantum EEG Simulation with GPU & Mixed Precision Optimization

## 📌 Introduction
### **Motivation & Inspiration**
The human brain exhibits **coherent oscillatory activity** across multiple spatial and temporal scales, forming what is observed as **EEG brainwaves**. These oscillations show remarkable self-organizing properties, coherence over long ranges, and fractal-like behaviors. Traditional models of EEG activity rely on **classical neural mass models**, but emerging evidence suggests that neural coherence phenomena bear striking **analogies to macroscopic quantum systems** such as **high-temperature superconductors**.

This project is inspired by:
1. **Scale Relativity Theory** – which provides a fractal spacetime structure to describe neural wavefunction dynamics.
2. **Pointer States in Quantum Mechanics** – EEG oscillations are modeled as stable quantum-like states that persist in neural dynamics.
3. **Long-Range Fractal Networks** – The brain's functional connectivity follows a scale-free, log-normal distribution akin to long-range correlations in complex systems like superconductors.

By leveraging these principles, we formulate EEG dynamics using a **modified Schrödinger equation**, incorporating an **effective neural potential** $V_{	ext{eff}}$ and a **macroscopic quantum potential (MQP)** $Q_{	ext{macro}}$ to model long-range coherence in brain oscillations.

---
## 🔬 **Mathematical Foundations**
### **1. Schrödinger-Like Evolution of EEG Oscillations**
To describe neural activity propagation, we use a **quantum-inspired wavefunction** $\Psi(x,t)$  evolving according to a modified Schrödinger equation:

$$
i \hbar \frac{\partial \Psi}{\partial t} = \left[ -\frac{\hbar^2}{2m} \nabla^2 + V_{\text{eff}} + Q_{\text{macro}} \right] \Psi
$$

where:
- **$V_{\text{eff}}(x,t)$** represents **log-normal distributed synaptic connectivity**, forming a complex neural landscape.
- **$Q_{\text{macro}}(x,t)$** is a **quantum-like potential** that stabilizes large-scale EEG coherence, analogous to quantum fluid dynamics.
- **$nabla^2$** models diffusion-like propagation across the cortical plane.

### **2. The Role of Scale Relativity in EEG Dynamics**
Scale Relativity (SR) extends classical physics by treating space and time as **fractal at small scales**, leading to stochastic and quantum-like dynamics in complex systems like the brain. Using SR principles, we derive a generalized neural diffusion equation:

$$
 i D \frac{\partial \, \Psi}{\partial t} = -D^2 \nabla^2 \Psi + V_{	ext{eff}} \Psi + Q_{	ext{macro}} \Psi
$$

where:
- **$D$** is a fractal diffusion coefficient governing neural excitability.
- The **fractal nature of neural activity** influences phase coherence in EEG oscillations.

### **3. Macroscopic Quantum Potential (MQP) & Long-Range Coherence**
In **superconductivity**, a macroscopic quantum potential **stabilizes Cooper pairs**, preventing decoherence. In EEG, we propose a similar mechanism where an MQP **stabilizes phase coherence in large-scale oscillations**:

$$
 Q_{\text{macro}}(x,t) = -\frac{\hbar^2}{2m} \frac{\nabla^2 \sqrt{\rho(x,t)}}{\sqrt{\rho(x,t)}} + \beta \sum_j W_j(x) \nabla^2 \rho_j(x,t)
$$

where:
- **$\rho(x,t) = |\Psi(x,t)|^2$** represents neural wave density.
- The **first term** is a Bohmian quantum potential, enforcing coherence.
- The **second term** models **log-normal synaptic connectivity**, ensuring long-range fractal coherence.

### **4. EEG & Superconductivity Analogy**
| **Feature** | **Quantum EEG Model** | **High-Temperature Superconductivity (HTSC)** |
|------------|----------------|-------------------------------|
| **Emergent Coherence** | EEG oscillations stabilize over large distances | Cooper pairs form long-range coherence |
| **Pointer States** | EEG frequency bands act as quantum pointer states | Superconducting state is a robust quantum phase |
| **Log-Normal Connectivity** | Synaptic strengths follow a log-normal law | Inhomogeneous pairing strength distribution |
| **Macroscopic Quantum Potential (MQP)** | Stabilizes EEG phase coherence | Enhances superconducting phase stability |
| **1/f Power Law** | EEG follows fractal scaling | Quantum fluctuations follow power-law behavior |

The **brain, like a superconductor, maintains macroscopic quantum-like coherence**, particularly during deep sleep states when MQP stabilizes EEG oscillations.

---
## 🚀 **Features & Modifications**
- **GPU-Accelerated Computation** using **JAX & CUDA**.
- **Crank-Nicholson Solver** for stable **wavefunction evolution**.
- **Log-Normal Synaptic Connectivity** to match real cortical networks.
- **Macroscopic Quantum Potential (MQP)** for large-scale coherence stabilization.
- **Mixed Precision (FP16 & FP32)** to **reduce VRAM usage** and improve efficiency.
- **Optimized for Large Grid Sizes** (256×256×128 resolution).
- **Parallelized EEG Analysis** (FFT, Spectral Entropy, PLI) for validation.

---
## ⚙️ Configuration & Hyperparameters
| Parameter | Description |
|-----------|-------------|
| GRID_SIZE | 256×256×128 (High-resolution cortical grid) |
| TIME_STEPS | 20000 (Long-term wave evolution) |
| DT | 0.0005 (Smaller time step for stability) |
| DIFFUSION_COEFFICIENT | 0.3 (Controls neural diffusion) |
| **Compute Optimizations** | **Status** |
| **GPU Acceleration** | ✅ Enabled (JAX CUDA) |
| **Mixed Precision** | ✅ Enabled (FP16 & FP32) |

---
## 🔧 **How to Run the Simulation**
1️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```
2️⃣ **Ensure GPU is enabled** (JAX CUDA backend)
```python
import jax
print(jax.devices())
```
3️⃣ **Run the simulation**
```bash
python main.py
```

---
## 🔥 **Future Enhancements**
- ✅ **Multi-GPU Support** for ultra-large simulations.
- ✅ **Hybrid Sleep-Wake Model** for transitioning EEG dynamics.
- ✅ **Real EEG Dataset Integration** for direct validation.

---
**Developed for cutting-edge quantum neuroscience and AI research! 🚀**
