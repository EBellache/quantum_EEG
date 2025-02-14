# Quantum EEG Simulation with GPU & Mixed Precision Optimization

## 📌 Introduction
### **Motivation & Inspiration**
The human brain exhibits **coherent oscillatory activity** across multiple spatial and temporal scales, forming what is observed as **EEG brainwaves**. These oscillations show remarkable self-organizing properties, coherence over long ranges, and fractal-like behaviors. Traditional models of EEG activity rely on **classical neural mass models**, but emerging evidence suggests that neural coherence phenomena bear striking **analogies to macroscopic quantum systems** such as **high-temperature superconductors** and **turbulent flows**.

This project is inspired by:
1. **Scale Relativity Theory** – which provides a fractal spacetime structure to describe neural wavefunction dynamics.
2. **Pointer States in Quantum Mechanics** – EEG oscillations are modeled as stable quantum-like states that persist in neural dynamics.
3. **Long-Range Fractal Networks** – The brain's functional connectivity follows a scale-free, log-normal distribution akin to long-range correlations in complex systems like superconductors and turbulence.

By leveraging these principles, we formulate EEG dynamics using a **modified Schrödinger equation**, incorporating an **effective neural potential $V_{	ext{eff}}(x,y,k,t)$** and a **macroscopic quantum potential (MQP) $Q_{	ext{macro}}(x,y,k,t)$** to model long-range coherence in brain oscillations.

---
## 🔬 **Mathematical Foundations**
### **1. Schrödinger-Like Evolution of EEG Oscillations**
**Scale Relativity** (SR) extends classical physics by **treating spacetime as fractal** at small scales, leading to the derivation of stochastic and quantum-like dynamics without the need for any ad-hoc postulate. This theory even extends to complex systems like the brain that often exhibit fractal signatures in their parameter space. 

Leveraging SR principles to describe neural activity propagation, we use a **quantum-inspired wavefunction** $\Psi(x,y,k,t)$, evolving according to a generalized Schrödinger equation:

$$
i\hbar \frac{\partial \Psi}{\partial t} = -\frac{D}{2} \nabla^2 \Psi + (V_{\text{eff}} + Q_{\text{macro}}) \Psi
$$

where:
- $\Psi(x,y,k,t)$ is the **neural wavefunction** on a **cortical grid**.
- $D$ is the **macroscopic diffusion constant** modeling neuronal excitability.
- $\nabla^2 = \nabla^2_x + \nabla^2_y + \frac{\partial^2}{\partial k^2}$ includes **spatial and scale resolution derivatives**.
- $V_{\text{eff}}$ represents the **effective excitation-inhibition potential** governing cortical interactions.
- $Q_{\text{macro}}$ is the **macroscopic quantum potential** enforcing large-scale coherence.

---

### **2 The Role of the scaling variable $k$ in EEG Simulation**
**From the perspective of Scale Relativity, $k$ serves as an additional coordinate encoding the resolution at which we probe the fractal space of neural activity**

✅ **$k$ represents the resolution at which we observe synaptic connectivity**  
✅ **$k$ introduces a structured connectivity space, influencing how EEG modes form and stabilize.**  
✅ **$k$-dependent Laplacian terms regulate neuronal coherence, affecting EEG frequency band separation**  
✅ **The Macroscopic Quantum Potential integrates $k$-dependent effects, ensuring global EEG synchronization.**  

---

### **3. Effective Potential $V_{\text{eff}}$**

An **effective potential** is derived from **neuronal excitation and inhibition interactions** and can be expressed in its continuous form as an **integral over synaptic interactions**:

$$
V_{\text{eff}}(x,y,k) = \int \int W_{\text{eff}}(x, y, x', y', k) |\Psi(x',y',k)|^2 \, dx' \, dy'
$$

where:
- $W_{\text{eff}}(x,y,x',y',k)$ represents the **continuous synaptic connectivity kernel**, governing excitation-inhibition interactions across the cortical surface.
- $|\Psi(x',y',k)|^2$ is the **neuronal activation probability density**, describing how activity propagates through synaptic interactions.
- $k$ encodes the **hierarchical inhibition-excitation balance**, influencing large-scale neural dynamics.

To implement $V_{\text{eff}}$ in a computationally efficient manner, we discretize the integral over the cortical grid **using a summation over neuronal sites**:

$$
V_{\text{eff}}(x,y,k) \approx \sum_{i', j'} W_{\text{eff}}(x, y, i', j', k) |\Psi(i',j',k)|^2
$$

where:
- The integral has been **approximated as a sum** over **discrete cortical sites** $(i', j')$.
- $W_{\text{eff}}(x, y, i', j', k)$ is now treated as a **log-normal distributed weight matrix**, encoding neuronal connectivity in a fractal network.
- This **discrete formulation** enables efficient computation,as well as approximating long-range interactions.

**Key Role:**  
$V_{\text{eff}}$ **stabilizes neural oscillations, enforces frequency coupling, and encodes external task loads**, ensuring that EEG frequency bands emerge as stable eigenstates of the system.

---
### **4. Expression for $Q_{\text{macro}}$ (Macroscopic Quantum Potential)**
The **macroscopic quantum potential** enforces **global neuronal coherence and EEG quantization**:

$$
Q_{\text{macro}} = -D \frac{\nabla^2 |\Psi|}{|\Psi|}
$$

where:
- $Q_{\text{macro}}$ **emerges from scale relativity** and **prevents uncontrolled decoherence**.
- The **second derivative of probability density** $\frac{\nabla^2 |\Psi|}{|\Psi|}$ ensures **self-organization of EEG modes**.
- **Large-scale cortical activity behaves like a resonant quantum-like system**.

**Key Role:**  
$Q_{\text{macro}}$ **introduces EEG frequency quantization, enforces large-scale coherence, and helps model brain-state transitions**.

---

### **5. Log-Periodicity of Neuronal Connections**
The **log-periodic structure** of neuronal connectivity emerges from **fractal organization of synaptic weights**. This follows a **log-normal distribution**:

$$
P(W) \sim \frac{1}{W} e^{-\frac{(\log W - \mu)^2}{2\sigma^2}}
$$

where:
- $W$ is the **synaptic weight**.
- $\mu$ and $\sigma$ define the **log-normal statistical structure** of neuronal connections.

🔹 **Key Implications:**
1. **Neuronal connectivity is neither fully random nor uniform but follows a log-periodic structure.**
2. **This fractal organization is responsible for EEG 1/f power-law scaling.**
3. **Log-periodicity introduces discrete EEG frequency bands, analogous to quantum energy levels.**

📌 **Why This Matters:**  
- **EEG power spectra exhibit a fractal 1/f distribution, suggesting scale-invariance in brain activity.**
- **The log-periodic correction in synaptic interactions leads to EEG frequency quantization.**
- **This structure supports hierarchical information processing and task-induced coherence.**

---

### **6. Biological Basis for Boundary Conditions**

#### Thalamocortical Filtering and Sleep-Wake Transitions
The boundary conditions in our model are inspired by the **role of the thalamus** in regulating cortical activity. The thalamus acts as a **sensory gate**, dynamically controlling how much neural energy flows between the cortex and the rest of the nervous system. This process directly influences **EEG wave coherence** and can be mathematically represented as a modulation of **wavefunction boundary absorption**.

- **Wakefulness:** The thalamus allows significant sensory input and energy dissipation, preventing large-scale cortical synchronization. This is modeled using **absorbing boundary conditions**, ensuring **localized wave propagation**.
- **Deep Sleep (NREM 3-4):** The thalamus **blocks external input**, forcing the cortex into an **isolated resonance state**, leading to **large, synchronized slow waves (0.5-4 Hz)**. This is effectively modeled by **reducing absorption**, allowing EEG waves to behave as in a **resonant cavity**.
- **REM Sleep:** The thalamus partially reopens, allowing **internally generated bursts of activity**, leading to theta waves (4-8 Hz) with intermittent desynchronization. This corresponds to **intermediate absorption strength**.

#### Mathematical Representation of Boundary Conditions
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

#### Implementation in Schrödinger Evolution
We apply this dynamic boundary condition in the wavefunction update step:

 $$
 \Psi_{t+1} = \Psi_t + dt \left( -D \nabla^2 \Psi_t + (V + Q_{\text{macro}}) \Psi_t \right) \cdot A(x, y, r, t)
 $$

This ensures that **energy remains trapped inside the cortex during sleep**, leading to large-scale EEG oscillations, while **wakefulness allows interaction with the rest of the nervous system**, preventing global coherence.

This biologically motivated boundary model provides a natural explanation for EEG transitions without requiring discrete switching between boundary conditions.

---
---

## **2 🔬 What Are We Testing?**
### **1️⃣ EEG Frequency Quantization**
- Can the **Schrödinger equation predict discrete EEG bands** (delta, theta, alpha, beta, gamma)?
- Does **wavefunction collapse under external task loads** correspond to EEG band locking?

### **2️⃣ 1/f Power Law & Fractal Scaling**
- Does **the power spectrum of EEG signals follow the expected 1/f trend**?
- How does **log-periodic connectivity affect fractal EEG dynamics**?

### **3️⃣ Sleep-Wake Transitions & Boundary Effects**
- Do **thalamic boundary conditions correctly model sleep-related EEG changes**?
- Does **deep sleep enhance neuronal coherence** via boundary confinement?

### **4️⃣ External Task Influence on Brain State**
- Does **introducing external input force the wavefunction into a particular EEG band**?
- Can we model **cognitive load as quantum measurement collapse**?

---

## **3 🔬 Our framework may extend to non vertebrates: The hydra example**

This example is based on findings from [6], which investigates the **bioelectric patterns** of *Hydra oligactis* in two conditions:  
1. **Immortal Hydra** (22°C) – which show **no signs of senescence** and maintain stable bioelectric gradients.  
2. **Mortal Hydra** (10°C) – which undergo **aging and loss of bioelectric coherence** over time.  

### **Key Findings**
- Immortal *Hydra* maintain a **stable foot-to-body voltage gradient**, exhibiting **persistent, bioelectric pointer states**.
- Aging mortal *Hydra* show a **progressive loss of bioelectric structure**, with **weaker, disorganized gradients**.
- The **breakdown of bioelectric coherence** correlates with **functional decline**, including **reduced contractility and feeding ability**.
- This suggests that **aging is associated with bioelectric decoherence**, similar to **loss of quantum pointer states in open quantum systems**.

### **Connection to EEG and Quantum Coherence**
The **bioelectric stability in immortal Hydra** can be interpreted as a **biological pointer state**, where a well-defined bioelectric field maintains **long-range coherence**, preventing random fluctuations. Conversely, the **aging process resembles decoherence**, as the bioelectric potential diffuses, leading to **functional decline and increased entropy**.  

This finding aligns with our **quantum EEG model**, where:
1. **Stable EEG frequency bands** represent **long-lived pointer states**, analogous to the **bioelectric stability of immortal Hydra**.
2. **Cognitive tasks and external stimuli induce wavefunction collapse**, forcing the brain into specific oscillatory modes.
3. **Sleep plays a crucial role in resetting neural coherence**, akin to how immortal Hydra **continuously maintain their bioelectric gradient**.

### **Broader Implications**
- **Aging may be interpreted as a progressive loss of coherence in a complex bioelectric field**, much like **quantum wavefunction decoherence**.
- **EEG quantization and frequency locking could be essential for stabilizing neural pointer states**, just as **Hydra maintain bioelectric gradients for survival**.
- **Sleep functions as a bioelectric reset mechanism**, helping restore coherence in brain activity, preventing the chaotic spread of neuronal fluctuations.

These findings suggest that **long-range coherence mechanisms**—whether in **neuronal oscillations, bioelectric patterns, or quantum systems**—are fundamental to sustaining **functional stability in living organisms**.  


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


