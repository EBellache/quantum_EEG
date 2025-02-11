# Quantum Recurrent Spiking Neural Network (QRSNN) EEG Simulation

## **📌 Introduction**
### **Motivation**
Understanding the nature of brain oscillations and their role in cognition is one of the fundamental challenges of neuroscience. EEG signals, which reflect the collective electrical activity of neuronal populations, exhibit characteristic frequency bands (Delta, Theta, Alpha, Beta, Gamma). One of the most intriguing properties of EEG power spectra is the presence of a **1/f power law**, indicative of fractal dynamics. Classical models based on stochastic processes and neural mass models capture some EEG properties but often fail to explain the **emergence of self-organized, quantized brainwave oscillations**. 

### **A New Approach: Quantum Neural Dynamics from Scale Relativity**
Recent advances in theoretical physics suggest that **Scale Relativity (SR)**, a generalization of relativity to non-differentiable (fractal) space-time, provides a robust framework to describe systems exhibiting **long-range correlations, self-organization, and power-law scaling**. This project applies the principles of **Scale Relativity** to model EEG oscillations as emerging from **quantum-like neural wave dynamics**. 

Our approach treats the **macroscopic activity of neuronal populations** as evolving according to a **Schrödinger-like equation**, derived directly from the **non-differentiable fractal geometry** of neural space-time. This allows for:
- **Emergent EEG frequency quantization** akin to eigenmodes in quantum systems.
- **A natural explanation for the 1/f power law** in EEG spectra.
- **Macroscopic Quantum Potential (MQP)-induced wave coherence**, which stabilizes oscillatory brain states.

By leveraging **Scale Relativity**, we propose that EEG oscillations are the **macroscopic signature of a deeper, scale-invariant neural process**, which can be described using quantum-like equations. 

---

## **📌 Parameter Space and Boundary Conditions**

### **1. Parameter Space**
The simulation models EEG activity within a **3D parameter space** consisting of:
- **(x, y)**: Spatial coordinates representing cortical surface locations.
- **r**: Neural firing rate space, capturing frequency-dependent oscillatory behavior.
- **t**: Time evolution of EEG activity.

This choice enables the modeling of **spatiotemporal EEG wave propagation and frequency band formation**.

### **2. Boundary Conditions**
- **Periodic Boundaries (x, y)**: Ensures continuous wave propagation across the cortical domain, preventing artificial edge effects.
- **Reflective Boundaries (r)**: Models neural activity confinement within biologically plausible firing rate ranges, supporting EEG frequency quantization.

These conditions allow for **natural EEG mode formation and self-organization** within the simulated cortical space.

---

## **📌 Simulation Framework**

The fundamental equation governing the model is:

$$
i D \frac{\partial \psi}{\partial t} = - D^2 \nabla^2 \psi + V_{\text{eff}} \psi + Q_{\text{macro}} \psi
$$

where:
- $\psi(x,y,r,t)$ is the **neural wavefunction**.
- $D$ is the **diffusion coefficient**, linked to EEG bands.
- $V_{\text{eff}}$ represents an **effective neural potential** incorporating excitation and inhibition.
- $Q_{\text{macro}}$ is the **Macroscopic Quantum Potential (MQP)**, ensuring wave coherence.

### **1. Effective Neural Potential**
The neural potential includes both **excitatory and inhibitory contributions**, defined as:

$$
V_{\text{eff}}(x,y,r,t) = V_{\text{exc}}(x,y,r,t) - g_{\text{inh}}(x,y,r,t) \psi
$$

where:
- $V_{\text{exc}}(x,y,r,t)$ represents the **excitatory synaptic input from pyramidal neurons**.
- $g_{\text{inh}}(x,y,r,t)$ is the **inhibitory coupling strength**, which regulates oscillatory stability.
- $\psi$ represents the **neural wavefunction**, linking inhibition to the global network state.

### **2. Macroscopic Quantum Potential (MQP)**
The MQP is derived from **Scale Relativity**, ensuring self-organization of oscillatory activity:

$$
Q_{\text{macro}} = - \frac{2 D^2}{m} \frac{\nabla^2 \sqrt{P}}{\sqrt{P}}
$$

where:
- $P = |\psi|^2$ is the **probability density of neural activity**.
- $D$ is the **diffusion constant**.
- $m$ is a mass-like parameter governing wave stability.

This term introduces **nonlinear self-regulation**, ensuring **EEG wave quantization** and **preventing uncontrolled diffusion**.

---

## **📌 How to Run the Simulation**

### **1️⃣ Install Dependencies**
```bash
pip install jax jaxlib numpy scipy matplotlib tkinter
```

### **2️⃣ Launch the Simulation**
```bash
python main.py
```

### **3️⃣ Expected Output**
- **Real-time EEG frequency spectrum visualization**.
- **Dynamic transition between EEG bands**.
- **Quantized oscillatory patterns emerging from the simulation**.

---

## **📌 Conclusion**
The **Scale Relativity approach** to EEG modeling provides:
- **A unified explanation for EEG quantized bands and self-organized oscillations**.
- **A natural derivation of the 1/f power law** as a fractal emergent property.
- **A link between macroscopic neural waves and fundamental scale-invariant physics**.

Future work will focus on:
- **Coupling this model with real EEG datasets**.
- **Exploring synaptic plasticity effects within the framework**.
- **Applying quantum-inspired techniques to neural computation and AI**.

