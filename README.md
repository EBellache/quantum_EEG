# **Quantum EEG Simulation**
🚀 **Exploring the Quantum-Like Nature of Brainwaves using Macroscopic Quantum Hydrodynamics**



## **📌 Project Overview**
This project aims to **simulate the emergence of quantized EEG frequency bands** using **quantum-inspired hydrodynamic equations** in a **fractal cortical network**. Leveraging **Scale Relativity Theory**, **log-normal neural connectivity**, and **JAX-based numerical methods**, we investigate whether EEG bands are the result of a **quantization principle** rather than classical oscillations.

### **🔬 Key Scientific Questions**
1. Are **EEG bands** merely classical oscillations, or do they reflect an **underlying quantum-like principle**?
2. Can **scale relativity** and **fractal cortical networks** explain **EEG frequency quantization**?
3. Is **sleep** a self-organizing process that **restores macroscopic quantum coherence** in the brain?
4. Can we **simulate large-scale EEG coherence** using **macroscopic quantum hydrodynamics**?

---

## **🧠 What Are We Testing?**
✅ **EEG Band Quantization** – Using a **Schrödinger-like equation for scale dynamics**, we explore how discrete EEG bands emerge.  
✅ **Fractal Cortical Connectivity** – Connections in the brain follow a **log-normal distribution**; we test how this shapes EEG structure.  
✅ **Macroscopic Quantum Potential** – EEG waves may be **stabilized by an emergent quantum potential**.  
✅ **Sleep as a Coherence Restorer** – Investigating whether sleep **reorganizes neural dynamics** to maintain EEG coherence.  
✅ **External Modulation of EEG** – Testing if **external neurostimulation** can shift EEG bands in a controlled manner.

---

## **⚡ Features**
- **Euler-Schrodinger Hydrodynamics**: Brainwaves simulated as a **quantum-like fluid**.
- **Log-Normal Neural Connectivity**: Cortical neurons are connected via a **log-normal synaptic weight matrix**.
- **Macroscopic Quantum Potential (MQP)**: EEG bands emerge from a **scale-dependent potential field**.
- **Multi-GPU/CPU Optimized**: Runs on **JAX**, fully **parallelized for RTX GPUs & multi-core CPUs**.
- **Fractal Network Dynamics**: EEG emerges from a **self-similar neural structure**.

---

## **🧑‍🔬 Scientific Background**
### **Macroscopic Quantum Hydrodynamics & EEG**
- **Macroscopic Quantum Potential (MQP)** explains why EEG bands appear as **quantized modes**.
- Cortical networks are **fractal** and follow a **log-normal connectivity pattern**.
- Sleep may restore **quantum coherence** by reorganizing EEG states.

  ---
  
### **Neural Connectivity and Fractal EEG Structures**
- Cortical networks **do not have uniform connectivity** but follow a **log-normal distribution**.
- Synaptic strengths between neurons are highly **skewed**, meaning **a few strong connections** dominate **a sea of weak ones**.
- This leads to the **self-organization of EEG bands** into **fractal, scale-dependent structures**.

Using **Scale Relativity Theory**, we propose that EEG bands are **not arbitrarily defined** but emerge from **quantized neural oscillatory states**.

---

### **Euler-Schrodinger Hydrodynamic Equations**
Instead of solving the classical Schrödinger equation:

$$
i \hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \psi + V \psi
$$

we use its **hydrodynamic representation**:

#### **1. Continuity Equation (Conservation of Probability Density)**
The probability density $\rho = |\psi|^2$ evolves according to:

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
$$

where $\mathbf{v}$ is the velocity field given by:

$$
\mathbf{v} = \frac{\nabla S}{m}
$$

with $S$ being the phase of the wavefunction.

#### **2. Euler-Like Equation (Quantum Force Balance)**
The velocity field obeys a modified Euler equation:

$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla) \mathbf{v} = -\frac{1}{m} \nabla V - \frac{1}{m} \nabla Q
$$

where:
- $V(x, y, s)$ is an **external neural potential** (e.g., synaptic inputs, neurotransmitter fields).
- $Q(x, y, s)$ is the **Macroscopic Quantum Potential (MQP)**.
  
---

### **Macroscopic Quantum Potential (MQP)**
The MQP is derived as:

$$
Q = -\frac{\hbar^2}{2m} \frac{\nabla^2 \sqrt{\rho}}{\sqrt{\rho}}
$$

This term plays a crucial role in stabilizing **coherent EEG states**, ensuring that frequency bands form **discrete, quantized modes**.


## **🧠 Updated Effective Neural Potential $V_{\text{eff}}$ with Excitation-Inhibition Balance**
This project models EEG band dynamics using an **effective neural potential** that incorporates:
1. **Log-normal cortical connectivity**
2. **Excitation-Inhibition balance across neurons**
3. **Neuronal activation probability**

### **📌 Updated Expression for $V_{\text{eff}}$**

$$
V_{\text{eff}}(x,y) = \sum_{i', j'} W_{\text{eff}}(x, y, i', j') E_{\text{eff}}(i',j') \rho(i',j')
$$

where:
- $W_{\text{eff}}(x,y, i', j')$ represents **log-normal synaptic connectivity weights**.
- $E_{\text{eff}}(i',j')$ = **Excitatory-Inhibitory balance factor** (+1 for excitatory, -1 for inhibitory).
- $\rho(i',j')$ = **Neuronal activation probability (EEG energy density).**

---

### **🔬 Biological Interpretation**
- **Excitatory neurons** increase EEG activation **(+1 effect)**.
- **Inhibitory neurons** reduce EEG activation **(-1 effect)**.
- **Log-normal connectivity ensures realistic cortical interactions**.
- **E/I balance evolves dynamically based on brain state.**

---

## **🧠 Thalamic Gating and Adaptive Boundary Conditions**
The **thalamus** acts as a **sensory gate** between the **cortex** and the **rest of the nervous system**. Depending on the **cortical state**, the **thalamus either blocks or allows information flow**.

This project implements **adaptive boundary conditions** that model **thalamic gating** as a **state-dependent absorption factor** $\alpha_{\text{thalamus}}$

---

### **📌 Boundary Conditions by Cortical State**
| **Cortical State**  | **α_thalamus Value** | **Boundary Condition** | **Expected EEG Behavior** |
|---------------------|--------------------|---------------------|---------------------|
| **Wakefulness**    | \(0.0\) (No absorption) | **Periodic (open)** | **EEG waves propagate freely**, mimicking real-time sensory processing. |
| **Light Sleep**    | \(0.3\) (Partial absorption) | **Mixed** | Some **damping at edges**, reducing sensory influence. |
| **Deep Sleep**     | \(0.8\) (Strong absorption) | **Closed-box resonance** | Strong **containment of activity** in the cortex (resonant EEG state). |
| **REM Sleep**      | \(0.2\) (Weak absorption) | **Localized damping** | Allows **localized activity** but prevents **global wave spread**. |

---

### **📌 Biological Rationale: How the Thalamus Controls EEG Waves**
- **Wakefulness** → The **thalamus is fully open**, allowing **EEG waves to spread** across the brain.
- **Light Sleep (Stage 1-2)** → The **thalamus starts blocking input**, partially **reducing EEG wave energy**.
- **Deep Sleep (Stage 3-4)** → The **thalamus fully blocks external input**, forcing the cortex into a **resonance box**, producing **Delta waves**.
- **REM Sleep** → The **thalamus selectively allows localized activity**, restricting **long-range propagation**.

---

### **📊 Expected Simulation Behavior**
✅ EEG band quantization **emerges naturally** from excitation-inhibition interactions.  
✅ **Cortical state transitions (Wakefulness vs. Sleep) modulate EEG activity.**  
✅ **Macroscopic Quantum Potential ensures stable EEG band structure.**  

🚀 **Now EEG waves are fully described by synaptic interactions & macroscopic quantum dynamics!**

---
### **📌 How to Run the EEG Simulation with Different States**
```bash
python -m src.simulations.eeg_band_quantization --state deep_sleep

---


### **Sleep and Quantum Coherence Restoration**
- Sleep is hypothesized to act as a **reset mechanism**, reorganizing EEG states.
- The presence of **slow-wave sleep (SWS) and REM cycles** suggests that **EEG coherence is periodically restored**.
- In this framework, **sleep acts as a macroscopic process that maintains neural coherence by shifting probability distributions in scale space**.

---

### **Implications for Neuroscience**
This **macroscopic quantum hydrodynamic model** allows us to:
1. **Explain EEG Band Quantization** – EEG bands arise **naturally** from an **effective potential landscape**.
2. **Understand Sleep Dynamics** – Sleep **resets** neural coherence through a **self-organizing cycle**.
3. **Develop EEG-Based Brainwave Modulation** – External stimulation may **shift EEG bands in a predictable manner**.

🚀 **This new approach bridges quantum mechanics, hydrodynamics, and neuroscience, offering a deeper understanding of brainwave coherence.**
