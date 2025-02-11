# Quantum Recurrent Spiking Neural Network (QRSNN) EEG Simulation

## **📌 Project Overview**
This project implements a **Quantum Recurrent Spiking Neural Network (QRSNN) EEG simulation** using **JAX**. The model is based on the principles of **Scale Relativity**, treating neural wave activity as a quantum-like process. This allows us to study the emergence and quantization of **EEG brainwave frequency bands (Delta, Theta, Alpha, Beta, Gamma)** while maintaining a balance between **excitation and inhibition (E/I balance)**.

The simulation is implemented on a **3D spatial-firing rate-time grid (x, y, r, t)** using:
- **Crank-Nicholson Method** for solving the **Neural Schrödinger Equation**.
- **Macroscopic Quantum Potential (MQP)** to stabilize wave coherence.
- **Excitatory and Inhibitory Synaptic Modeling** to regulate neural oscillations.
- **Hybrid Boundary Conditions**: **Periodic (x, y)** and **Reflective (r)** for EEG wave quantization.
- **Fast Fourier Transform (FFT)** for EEG spectral analysis.
- **Tkinter GUI** for real-time EEG visualization.

---
## **🧠 Scientific Background**

### **1. Quantum Formalism for Neural Dynamics**
Traditional spiking neural networks (SNNs) describe neuronal activity using **leaky integrate-and-fire (LIF) models**:

$$
\[
\tau_m \frac{dV_i}{dt} = - (V_i - E_L) + I_i^{\text{exc}}(t) - I_i^{\text{inh}}(t)
\]
$$

where:
- $V_i$ is the **membrane potential** of neuron $i$,
- $\tau_m$ is the **membrane time constant**, 
- $E_L$ is the **resting potential**, 
- $I_i^{\text{exc}}(t)$ is the **excitatory synaptic input**,
- $I_i^{\text{inh}}(t)$ is the **inhibitory synaptic input**.

Scale Relativity extends this by introducing **fractal space-time fluctuations**, leading to a **Schrödinger-like equation** for neuronal wave dynamics:

\[

i D \frac{\partial \psi}{\partial t} = -D^2 \nabla^2 \psi + V_{\text{eff}} \psi + Q_{\text{macro}} \psi
\]

where:
- $\psi(x,y,r,t)$ is the **neural wavefunction** representing probability amplitudes in **spatial-firing rate-time space**.
- $D$ is the **diffusion coefficient** (analogous to \( \hbar/2m \) in quantum mechanics).
- $V_{\text{eff}}$ represents an **effective neural potential**, which now includes both excitatory and inhibitory terms:
 $$
  V_{\text{eff}}(x,y,r,t) = V_{\text{exc}}(x,y,r,t) - g_{\text{inh}}(x,y,r,t) \psi
  \]
  where $g_{\text{inh}}$ represents the **strength of inhibition**.
- $Q_{\text{macro}}$ is the **Macroscopic Quantum Potential (MQP)**, which governs self-organizing neural coherence.

### **2. EEG Frequency Band Quantization with Excitation-Inhibition Balance**
In human EEG, distinct frequency bands emerge naturally:

| **Band** | **Frequency (Hz)** | **Function** |
|---------|-----------------|------------------|
| **Delta** | 0.5 – 4 Hz | Deep sleep, unconscious processing |
| **Theta** | 4 – 8 Hz | Memory formation, deep relaxation |
| **Alpha** | 8 – 12 Hz | Resting state, relaxation |
| **Beta** | 12 – 30 Hz | Active thinking, problem-solving |
| **Gamma** | 30+ Hz | High-level cognition, consciousness |

Inhibition is critical for the regulation of these frequencies:
- **High inhibition** enhances **Gamma waves** while suppressing **slow-wave oscillations**.
- **Low inhibition** promotes **Delta/Theta activity** but may reduce cognitive processing.

The model **dynamically balances excitation and inhibition** to simulate EEG-like behavior.

---
## **🔬 Methodology**
### **1. Numerical Solver: Crank-Nicholson Method with Inhibitory Modulation**
The Crank-Nicholson scheme is used for time evolution:
\[
\psi^{n+1} = \psi^n + \frac{i \Delta t}{D} \left(-D^2 \nabla^2 \psi + V_{\text{eff}} \psi + Q_{\text{macro}} \psi \right)
\]
where $V_{\text{eff}}$ now includes **excitatory-inhibitory balance**.

### **2. Hybrid Boundary Conditions**
| **Dimension** | **Boundary Condition** | **Effect** |
|--------------|---------------------|-------------|
| **(x, y)** | **Periodic** $\psi(x+L, y+L) = \psi(x,y)$ | Global EEG wave coherence |
| **(r)** | **Reflective** $\frac{\partial \psi}{\partial r} \bigg|_{0,R} = 0$ | Quantization of EEG bands |

### **3. Fast Fourier Transform (FFT) for EEG Analysis**
\[
\mathcal{F}\{\psi(x,y,r,t)\} = \sum_{x,y,r} \psi(x,y,r,t) e^{-i2\pi ft}
\]

This extracts EEG **frequency spectra**, showing dynamic transitions between brainwave bands.

### **4. Real-Time EEG Visualization**
- **Tkinter GUI** visualizes **EEG power spectra**.
- **FFT heatmaps** track **band transitions dynamically**.

---
## **⚡ Installation & Execution**
### **1️⃣ Install Dependencies**
```bash
pip install jax jaxlib numpy scipy matplotlib tkinter
```
### **2️⃣ Run the Simulation**
```bash
python main.py
```
### **3️⃣ Expected Output**
- **Real-time EEG frequency spectrum visualization**.
- **Dynamic transition between EEG bands**.

---
## **🚀 Future Enhancements**
- **Neural plasticity modeling** via dynamic synaptic weights.
- **Coupling with real EEG datasets** for validation.
- **Integration with deep learning** for hybrid quantum neural decoding.

---
## **📜 References**
1. Nottale, L. (1993). *Fractal Space-Time and Microphysics: Towards a Theory of Scale Relativity.*
2. Deco, G., & Jirsa, V. K. (2012). "Ongoing Cortical Activity at Rest: Criticality, Multistability, and Ghost Attractors." *Journal of Neuroscience*.
3. Freeman, W. J. (2003). "Evidence from Human EEG for Network Dynamics of Consciousness." *Neural Networks*.

---
## **🎯 Contribution & License**
- **Developers**: Bellachehab
- **License**: MIT License
- Contributions are welcome! Fork the repository and submit pull requests.

