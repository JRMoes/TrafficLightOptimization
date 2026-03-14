# Traffic Light Optimization using SPSA

This repository contains a simulation and optimization framework for **traffic light control at a four-way intersection**.
The project was developed for an **Operations Research case study**, where the objective is to reduce the **average time vehicles spend at the intersection** using a **data-driven traffic light controller** optimized with **Simultaneous Perturbation Stochastic Approximation (SPSA)**.

The system combines:

* A **Discrete Event Simulation (DES)** of traffic flow
* A **Self-Organizing Traffic Light (SOTL) controller**
* A **stochastic optimization algorithm (SPSA)** to learn optimal switching thresholds.

The implementation is written in **Python**.

---

# Project Overview

Traffic congestion is a major challenge in urban areas. Traditional traffic light systems often use fixed cycles that do not adapt to real-time traffic conditions.

This project implements an **adaptive traffic light controller** inspired by the self-organizing traffic lights method proposed by Cools et al. (2012), where each traffic light responds to local traffic demand.

Instead of fixed switching times, the controller measures the **integral of waiting vehicles over time**.
If this value exceeds a threshold θ, the signal switches.

The thresholds are **not chosen manually** but **learned automatically** using **SPSA optimization**.

---

# Model Description

The intersection consists of **four approaches**:

* `Ba1` – Bakeri 1
* `Ba2` – Bakeri 2
* `Be1` – Besat 1
* `Be2` – Besat 2

Each approach contains **three lanes**:

| Lane | Movement        |
| ---- | --------------- |
| 1    | Left / Straight |
| 2    | Left / Straight |
| 3    | Right turn      |

Vehicles arrive according to **empirically fitted probability distributions**, and each vehicle is assigned:

* a **turning direction**
* a **crossing time**
* a **lane choice**

The simulation tracks each vehicle from **arrival until exiting the intersection**.

Performance is measured using:

**Mean Time in System**

```
time_in_system = exit_time - arrival_time
```

---

# Self-Organizing Traffic Light Controller

The implemented controller follows the **SOTL concept**:

For each approach (i):

```
κ_i = integral of (number of cars waiting at red light)
```

If

```
κ_i ≥ θ_i
```

and the minimum green time has elapsed, the signal switches.

Key features implemented:

* Minimum green time (φ_min)
* All-red safety phase
* Lane-based queues
* Event-driven signal switching

This allows the traffic system to **adapt dynamically to traffic demand**.

---

# Optimization with SPSA

To determine good threshold values (θ), the project uses:

**Simultaneous Perturbation Stochastic Approximation (SPSA)**

SPSA estimates the gradient of the objective function using only **two simulations per iteration**, regardless of the number of parameters.

Optimization goal:

```
minimize mean_time_in_system(θ)
```

Update rule:

```
θ_{k+1} = θ_k − ε * ĝ_k
```

where the gradient estimate is computed using:

```
ĝ_k = (J(θ+ηΔ) − J(θ−ηΔ)) / (2η) * (1/Δ)
```

This allows efficient optimization despite the simulation being **stochastic and expensive to evaluate**.

---

# Repository Structure

```
.
├── SPSA TLO.py
└── README.md
```

### `SPSA TLO.py`

Main program containing:

* Traffic simulation engine
* SOTL controller implementation
* SPSA optimization
* Visualization and animation tools

---

# Running the Code

The behaviour of the program is controlled using the **CONFIG dictionary** at the bottom of `SPSA TLO.py`.

## 1. Run a Single Simulation

Set:

```python
"MODE": "batch"
```

Then run:

```bash
python "SPSA TLO.py"
```

Output:

* Mean time in system
* Mean time per street
* Queue lengths

---

## 2. Run the Animation

Set:

```python
"MODE": "animate"
```

This produces a **real-time visualization of the intersection**, including:

* vehicle queues
* signal states
* vehicle movements

Optional:

```python
"SAVE_MP4": True
```

to export a video.

---

## 3. Run the SPSA Optimization

Set:

```python
"MODE": "spsa"
```

The algorithm will:

1. Run repeated simulations
2. Estimate gradients using SPSA
3. Update threshold parameters
4. Plot optimization progress

Example parameters:

```python
"SPSA_THETA0": [60,60,60,60],
"SPSA_EPSILON": 5.0,
"SPSA_ETA": 30.0,
"SPSA_N_ITER": 200,
"SPSA_BATCH": 5
```

---

# Example Output

After optimization the program prints:

```
Optimised theta (Ba1,Ba2,Be1,Be2): [...]
Final estimated objective: ...
```

It also produces:

* Parameter convergence plots
* Objective value plots

---

# Dependencies

Install required Python packages:

```bash
pip install numpy scipy matplotlib
```

Optional (for video export):

```bash
pip install ffmpeg
```

---

# References

Kamran, M. A., et al. (2017).
*Traffic light signal timing using simulation.*

Cools, S. B., Gershenson, C., & D’Hooghe, B. (2012).
*Self-organizing traffic lights: A realistic simulation.*

---

# Contributors

This project was developed as part of an **Operations Research case study**.

<a href="https://github.com/JRMoes">
  <img src="https://github.com/JRMoes.png" width="60px" alt="Jesse Moestaredjo"/>
</a>
<a href="https://github.com/PSchep13">
  <img src="https://github.com/PSchep13.png" width="60px" alt="Teammate 2"/>
</a>
<a href="https://github.com/avandersluys">
  <img src="https://github.com/avandersluys.png" width="60px" alt="Teammate 1"/>
</a>

