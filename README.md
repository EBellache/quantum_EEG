# Quantum EEG Simulation with GPU & Mixed Precision Optimization

## 📌 Introduction
### **Motivation & Inspiration**
The human brain exhibits **coherent oscillatory activity** across multiple spatial and temporal scales, forming what is observed as **EEG brainwaves**. These oscillations show remarkable self-organizing properties, coherence over long ranges, and fractal-like behaviors. Traditional models of EEG activity rely on **classical neural mass models**, but emerging evidence suggests that neural coherence phenomena bear striking **analogies to macroscopic quantum systems** such as **high-temperature superconductors** and **turbulent flows**.

This project is inspired by:
1. **Scale Relativity Theory** – which provides a fractal spacetime structure to describe neural wavefunction dynamics.
2. **Pointer States in Quantum Mechanics** – EEG oscillations are modeled as stable quantum-like states that persist in neural dynamics.
3. **Long-Range Fractal Networks** – The brain's functional connectivity follows a scale-free, log-normal distribution akin to long-range correlations in complex systems like superconductors and turbulence.

By leveraging these principles, we formulate EEG dynamics using a **modified Schrödinger equation**, incorporating an **effective neural potential $V_{	ext{eff}}(x,y,r,t)$** and a **macroscopic quantum potential (MQP) $Q_{	ext{macro}}(x,y,r,t)$** to model long-range coherence in brain oscillations.

---
## 🔬 **Mathematical Foundations**
### **1. Schrödinger-Like Evolution of EEG Oscillations**
To describe neural activity propagation, we use a **quantum-inspired wavefunction** $\Psi(x,y,r,t)$, evolving according to a modified Schrödinger equation:

 $$
 i \hbar \frac{\partial \Psi}{\partial t} = \left[ -\frac{\hbar^2}{2m} (\nabla_{xy}^2 + \partial_r^2) + V_{\text{eff}}(x,y,r,t) + Q_{\text{macro}}(x,y,r,t) \right] \Psi
 $$

where:
- **$x, y$** are spatial cortical coordinates.
- **$r$** represents an **inhibition-excitation balance parameter**, encoding hierarchical neural states, synaptic conductance variations, and cross-frequency coupling effects.
- **$\nabla_{xy}^2 \Psi$** models the diffusion-like propagation of neural activity across the cortical sheet.
- **$\partial_r^2 \Psi$** governs transitions between nested frequency bands and **excitation-inhibition states**.
- **$V_{\text{eff}}(x,y,r,t)$** represents **log-normal distributed synaptic connectivity**, forming a complex neural landscape. Plus any tasks imposed by the exterior environement.
- **$Q_{\text{macro}}(x,y,r,t)$** is a **quantum-like potential** that stabilizes large-scale EEG coherence, analogous to quantum fluid dynamics.

### **2. The Role of Scale Relativity in EEG Dynamics**
Scale Relativity (SR) extends classical physics by treating space and time as **fractal at small scales**, leading to stochastic and quantum-like dynamics in complex systems like the brain. Using SR principles, we derive a generalized neural diffusion equation:

$$
 i D \frac{\partial \, \Psi}{\partial t} = -D^2 (\nabla_{xy}^2 + \partial_r^2) \Psi + V_{	ext{eff}}(x,y,r,t) \Psi + Q_{	ext{macro}}(x,y,r,t) \Psi
$$

where:
- **$D$** is a fractal diffusion coefficient governing neural excitability.
- The **fractal nature of neural activity** influences phase coherence in EEG oscillations.

### **3. Macroscopic Quantum Potential (MQP) & Long-Range Coherence**
In **superconductivity**, a macroscopic quantum potential **stabilizes Cooper pairs**, preventing decoherence. In EEG, we propose a similar mechanism where an MQP **stabilizes phase coherence in large-scale oscillations**:

$$
 Q_{\text{macro}}(x,y,r,t) = -\frac{\hbar^2}{2m} \frac{\nabla_{xy}^2 \sqrt{|\Psi(x,y,r,t)|^2}}{\sqrt{|\Psi(x,y,r,t)|^2}} 
$$

$$
 V(x,y,r,t)  = \beta \sum_j W_j(x,y) \nabla_{xy}^2 |\Psi_j(x,y,r,t)|^2
$$

where:
- **$|\Psi(x,y,r,t)|^2$** represents the neural wave density, encoding excitation-inhibition balance.
- The **first term** is a Bohmian quantum potential, enforcing coherence.
- The **second term** models **log-normal synaptic connectivity**, ensuring long-range fractal coherence.:
- The total **effective potential** is: $V_{\text{eff}}(x,y,r,t) = V(x,y,r,t) + V_{\text{external}}(x,y,r,t)$ with the external potential being optional and models any external cognitive load


### **4. EEG, Superconductivity & Turbulence Analogy**
| **Feature** | **Quantum EEG Model** | **High-Temperature Superconductivity (HTSC)** | **Turbulence (Pointer States Paper)** |
|------------|----------------|-------------------------------|-------------------|
| **Governing Equation**  | **Schrödinger-like Wave Equation** for neural activity | **Ginzburg-Landau Equations** for superconducting phase coherence | **Navier-Stokes Equation** with energy cascades |
| **Coherent Structures** | **EEG frequency bands (alpha, beta, gamma, delta)** | **Cooper pairs in superconducting phase** | **Vortices in turbulence** |
| **Decoherence Mechanism** | Wakefulness slowly disrupting neural coherence | Thermal fluctuations disrupting superconducting state | Viscous dissipation breaking turbulent structures |
| **Macroscopic Order** | Long-range EEG coherence in sleep | Quantum phase coherence in superconductors | Large-scale turbulence patterns |
| **Energy Exchange** | EEG cross-frequency coupling | Energy exchange between superconducting pairs | Kolmogorov cascade|

---
## **5 The Role of $r$ in EEG Simulation**
✅ **$r$ encapsulates the inhibition-excitation balance, capturing neural state transitions.**  
✅ **$r$ introduces hierarchical oscillatory states, leading to EEG frequency quantization.**  
✅ **$r$-dependent boundary conditions control wave stability, ensuring coherence in EEG bands.**  
✅ **The Macroscopic Quantum Potential stabilizes EEG eigenstates, preventing chaotic fluctuations.**  

By explicitly incorporating **$y$ (spatial diffusion) and $r$ (inhibition-excitation dynamics)**, our model accurately describes **EEG self-organization, quantization, and coherence phenomena.**

## **6 Biological Basis for Boundary Conditions**

### Thalamocortical Filtering and Sleep-Wake Transitions
The boundary conditions in our model are inspired by the **role of the thalamus** in regulating cortical activity. The thalamus acts as a **sensory gate**, dynamically controlling how much neural energy flows between the cortex and the rest of the nervous system. This process directly influences **EEG wave coherence** and can be mathematically represented as a modulation of **wavefunction boundary absorption**.

- **Wakefulness:** The thalamus allows significant sensory input and energy dissipation, preventing large-scale cortical synchronization. This is modeled using **absorbing boundary conditions**, ensuring **localized wave propagation**.
- **Deep Sleep (NREM 3-4):** The thalamus **blocks external input**, forcing the cortex into an **isolated resonance state**, leading to **large, synchronized slow waves (0.5-4 Hz)**. This is effectively modeled by **reducing absorption**, allowing EEG waves to behave as in a **resonant cavity**.
- **REM Sleep:** The thalamus partially reopens, allowing **internally generated bursts of activity**, leading to theta waves (4-8 Hz) with intermittent desynchronization. This corresponds to **intermediate absorption strength**.

### Mathematical Representation of Boundary Conditions
We define a **dynamic absorption mask** that smoothly transitions between these states:

 $$
 A(x, y, r, t) = \exp\left(-\left(\lambda_0 - \lambda_s S(t)\right) \left( \frac{x^2}{L_x^2} + \frac{y^2}{L_y^2} + \frac{r^2}{L_r^2} \right) \right)
 $$

where:
- $\lambda_0 = 0.95$ is the default absorption factor in **wakefulness**.
- $\lambda_s = 0.4$ modulates absorption strength based on sleep state.
- $S(t)$ is a **sleep factor** that transitions smoothly from 0 (wake) to 1 (deep sleep).
- $(L_x, L_y, L_r)$ are characteristic cortical grid scales.

This function controls **how much energy remains inside the cortex** vs. **how much is dissipated into the nervous system**.

### Implementation in Schrödinger Evolution
We apply this dynamic boundary condition in the wavefunction update step:

 $$
 \Psi_{t+1} = \Psi_t + dt \left( -D \nabla^2 \Psi_t + (V + Q_{\text{macro}}) \Psi_t \right) \cdot A(x, y, r, t)
 $$

This ensures that **energy remains trapped inside the cortex during sleep**, leading to large-scale EEG oscillations, while **wakefulness allows interaction with the rest of the nervous system**, preventing global coherence.

This biologically motivated boundary model provides a natural explanation for EEG transitions without requiring discrete switching between boundary conditions.


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
## **📌 References**
1. **Nottale, L.** "The Nature of Pointer States and Their Role in Macroscopic Quantum Coherence"- https://www.mdpi.com/2410-3896/9/3/29
2. **Buzsáki, G.** "Rhythms of the Brain" - https://academic.oup.com/book/11166
4. **Zurek, W. H.** "Preferred States, Predictability, Classicality and the Environment-Induced Decoherence" - https://academic.oup.com/ptp/article/89/2/281/1847355
5. **Ghose, P**. Derivation of a Schrodinger Equation for Single Neurons Through Stochastic Neural Dynamics - arXiv preprint arXiv:2406.16991.

These references form the theoretical foundation for **EEG quantization, macroscopic quantum effects, and scale relativity principles in neural dynamics**.


