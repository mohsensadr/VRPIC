![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Vlasov-Poisson with Variance Reduction

A CUDA/C++ implementation of the **particle-in-cell (PIC)** method of solving the **Vlasov–Poisson equation** equipped with the **Variance Reduction (VRPIC)**. This project provides a high-performance GPU-accelerated framework for simulating plasma dynamics with reduced statistical noise, and enabling more accurate long-time evolution of distribution functions.

---

## Overview

The **Vlasov–Poisson equation** describes the evolution of a plasma or charged particle system under self-consistent electric fields. Traditional particle-in-cell (PIC) methods can suffer from noise. The proposed method addresses this challenge by taking advantage of the correlation between the non-equilibrium and equilibrium simulations via  **importance weights**. For example, the density and temperature profiles for the Landau Damping test case at the finite time can be estimated with a lower variance compared to standard PIC.

![Demo](examples/LandauDamping.gif)

---

## Features

- **Fully GPU-accelerated**: Uses CUDA to parallelize moment computation, particle updates, and field (Poisson) solver.
- **Variance reduction (VR)**: Implements control variate methods to reduce noise in moment computations.
- **Importance weighting**: Dynamically adjusts particle weights using local Maxwellian-Boltzmann distribution as control variate.
- **Self-consistent field solving**: Solves the Poisson equation using FFT method.
- **Post-processing output**: Dumps moment fields for visualization and diagnostics.

---

## 🛠️ Build Instructions

### Requirements

- CUDA Toolkit (>= 11.x recommended)
- C++ compiler with C++11 or higher
- CMake

### Build

```bash
git clone https://github.com/yourusername/vlasov-poisson.git
cd vlasov-poisson
mkdir bin && cd bin
cmake ..
make
```

By default, cmake compile the code for A100 GPU. In casee of other architectures, provide cmake with the flag `CMAKE_CUDA_ARCHITECTURES`, for example
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
       [pdf_params...]
```
where 

```
deposition_mode: brute | tiling | sorting
VRMode: basic | MXE
RhsMode: MC | VR
```

For example:

``` ./main 100 100 1000000 0.1 200 12.5663706144 12.5663706144  256 sorting mxe vr cosine 0.05 0.5```

For the command line of executioning different test cases, see the header in ```src/main.cpp```.
