[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2602.15041.svg)](https://doi.org/10.48550/arXiv.2602.15041)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# VRPIC: Variance Reduction for Particle-in-Cell method

A CUDA/C++ implementation of the **particle-in-cell (PIC)** method of solving the **Vlasov–Poisson equation** equipped with the **Variance Reduction (VRPIC)**. This project provides a high-performance GPU-accelerated framework for simulating plasma dynamics with reduced statistical noise, and enabling more accurate long-time evolution of distribution functions. This git repository has been used to produce results in the following paper:

Victor Windhab, Andreas Adelmann, Mohsen Sadr. "VR-PIC: An entropic variance-reduction method for particle-in-cell solutions of the Vlasov-Poisson equation." 2026, preprint at [arXiv:2602.15041](https://doi.org/10.48550/arXiv.2602.15041).

---

## Overview

The **Vlasov–Poisson equation** describes the evolution of a charged particle system under self-consistent electric fields. Traditional particle-in-cell (PIC) methods suffer from inherent statistical noise when the signal is weak. The proposed method addresses this challenge by taking advantage of the correlation between the non-equilibrium and equilibrium simulations via  **importance weights**. For example, the density and temperature profiles for the Landau Damping test case at the finite time can be estimated with a lower variance compared to standard PIC.

![Demo](examples/LandauDamping.gif)

---

## Features

- **Fully GPU-accelerated**: Uses CUDA to parallelize moment computation, particle updates, and field (Poisson) solver.
- **Variance reduction (VR)**: Implements control variate methods to reduce noise in moment computations.
- **Importance weighting**: Dynamically adjusts particle weights using local Maxwellian-Boltzmann distribution as control variate.
- **Least biased moment conservation**: Deploys MxE formulation to ensure weight conservation during the kick process.
- **Self-consistent field solving**: Solves the Poisson equation using FFT method.
- **Post-processing output**: Dumps moment fields for visualization and diagnostics.
- **Weight diagnostic**: Records `step,time,max_weight,max_mxe_iterations` in `data/max_weight.csv` at the end of every time step.

---

## 🛠️ Build Instructions

### Requirements

- CUDA Toolkit (>= 11.x recommended)
- C++ compiler with C++11 or higher
- CMake

### Build

```bash
git clone https://github.com/mohsensadr/VRPIC.git
cd VRPIC
mkdir bin && cd bin
cmake ..
make
```

By default, cmake compile the code for A100 GPU. In case of other architectures, provide cmake with the flag `CMAKE_CUDA_ARCHITECTURES`, for example
```
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
```

### Execution

The compiled executable can be run by
```
./main N_GRID_X\
       N_GRID_Y\
       N_PARTICLES\
       CFL\
       NSteps\
       Lx\
       Ly\
       threadsPerBlock\
       deposition_mode\
       VRMode\
       RhsMode\
       [pdf_type]\
       [pdf_params...]\
       [--field-output on|off]
```
where 

```
deposition_mode: brute | tiling | sorting
VRMode: basic | MXE
RhsMode: MC | VR
```

For example:

``` ./main 100 100 1000000 0.1 200 12.5663706144 12.5663706144  256 sorting mxe vr cosine 0.05 0.5```

Field CSV output is enabled by default. For long simulations, disable the large
initial and periodic field dumps by placing this option after the PDF parameters:

```bash
./main 100 100 1000000 0.1 10000 12.5663706144 12.5663706144 256 sorting mxe vr cosine 0.05 0.5 --field-output off
```

This option does not disable `data/max_weight.csv`; the initial maximum weight
at step 0 and the maximum weight/maximum MxE iterations after every time step
continue to be recorded. Use
`--field-output on` to enable field output explicitly.

At the end of every successful run, VRPIC also replaces
`data/performance_metrics.csv`. The file records the particles per cell, total
particle count, wall-clock execution time, peak tracked GPU allocation in MiB,
and the same memory value in bytes. The GPU-memory value covers every direct
VRPIC CUDA allocation, including transient sorting, Poisson, and reduction
workspaces. It intentionally excludes CUDA-context and cuFFT-internal memory.

After preserving the three run directories as `bin/data_1e2`, `bin/data_1e3`,
and `bin/data_1e4`, generate charge-density accuracy-versus-cost figures with:

```bash
python3 examples/plot_vrpic_charge_error_cost.py
```

The plotting script reads `performance_metrics.csv` from each directory
automatically. Explicit `--metrics`, `--execution-times`, and
`--memory-footprints` options remain available for older runs.

Each run replaces `data/max_weight.csv`. Preserve or rename this file between
Landau-damping runs with different cosine amplitudes. Its time column uses the
simulation time `step * DT`, so runs can be compared directly even when their
time-step sizes differ.

To compare runs stored as `bin/data_alpha_<value>/max_weight.csv`, install
Matplotlib and generate the
maximum-weight and MxE-iteration figures with:

```bash
python3 examples/plot_landau_weight_diagnostics.py
```

The script writes single-column (3.5-inch-wide), publication-ready PDF and
600-DPI PNG files to `bin/figures`.
Maximum weight is plotted as $\|w(t)\|_\infty$ on a logarithmic vertical axis.
Both figures use distinct line styles and staggered markers spaced every 10,000
steps. MxE iterations use a logarithmic axis by default; use
`--iteration-scale linear` if a linear MxE iteration axis is preferred.

For the command line of executioning different test cases, see the header in ```src/main.cpp```.
