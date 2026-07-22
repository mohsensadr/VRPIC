#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <cuda_runtime.h>

#include "Constants/constants.hpp"
#include "Solvers/solver.cuh"
#include "Initializations/initialization.cuh"
#include "IOs/IO.h"
#include "IOs/MaximumWeightRecorder.cuh"
#include "Depositors/moments.cuh"
#include "Containers/particle_container.cuh"
#include "Containers/field_container.cuh"
#include "Distributions/pdfs.cuh"
#include "Sorters/sorting.cuh"
#include "VRs/MxE.cuh"

void run(const std::string& pdf_type, float_type* pdf_params,
         bool field_output_enabled) {
    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float_type));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float_type));

    // TODO: dx, dy, Lx, Ly are member variables of field container, remove them from here.
    dx = Lx/N_GRID_X;
    dy = Ly/N_GRID_Y;
    grid_size = N_GRID_X*N_GRID_Y;
    QP = Lx*Ly/N_PARTICLES;
    MP = 1.0;

    const bool mxe_enabled = vrMode == VRMode::MXE;
    const bool vr_enabled = rhsMode == RhsMode::VR || mxe_enabled;

    ParticleContainer pc(N_PARTICLES, vr_enabled, mxe_enabled);
    FieldContainer fc(N_GRID_X, N_GRID_Y, Lx, Ly, vr_enabled, mxe_enabled);
    Sorting sorter(pc, fc);

    // Create the appropriate PDF struct for device use
    PDF_position pdf_position;
    if (pdf_type == "gaussian" || pdf_type == "Gaussian") {
        pdf_position = make_gaussian_pdf(pdf_params[0], Lx, Ly);
    } else if (pdf_type == "cosine" || pdf_type == "Cosine") {
        pdf_position = make_cosine_pdf(pdf_params[0], pdf_params[1], Lx, Ly);
    } else if (pdf_type == "double_gaussian" || pdf_type == "DoubleGaussian") {
        pdf_position = make_double_gaussian_pdf(pdf_params[0], pdf_params[1], pdf_params[2], pdf_params[3], 
                                              pdf_params[4], pdf_params[5], pdf_params[6], pdf_params[7], Lx, Ly);
    } else {
        throw std::invalid_argument("Unknown PDF type: " + pdf_type);
    }

    // initialize particle velocity and position
    initialize_particles(pc, pdf_position);

    // compute moments, needed to find emperical density field
    if (depositionMode == DepositionMode::SORTING)
        sorter.sort_particles_by_cell();
    compute_moments(pc, fc, sorter);

    if (vr_enabled) {
        // Set particle weights and recompute the variance-reduced moments.
        initialize_weights(pc, fc, pdf_position);
        if (depositionMode == DepositionMode::SORTING)
            sorter.sort_particles_by_cell();
        compute_moments(pc, fc, sorter);
    }

    // compute Electric field
    solve_poisson_periodic(fc);

    // Field dumps can be disabled for long runs. Scalar diagnostics remain on.
    if (field_output_enabled)
        post_proc(fc, 0);

    const size_t size = N_PARTICLES * sizeof(float_type);
    std::unique_ptr<MaximumWeightRecorder> maximum_weight_recorder;
    if (vr_enabled) {
        maximum_weight_recorder =
            std::make_unique<MaximumWeightRecorder>(N_PARTICLES);
        // Preserve the true initial maximum for subsequent diagnostics.
        maximum_weight_recorder->begin_step();
        maximum_weight_recorder->record(0, 0.0, pc.d_w);
    }

    for (int step = 1; step < NSteps+1; ++step) {
        if (vr_enabled)
            maximum_weight_recorder->begin_step();

        // compute Electric field
        solve_poisson_periodic(fc);

        if (vr_enabled) {
            if (mxe_enabled) {
                cudaMemcpy(pc.d_wold, pc.d_w, size, cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();
            }
            // Map weights from global to local equilibrium.
            pc.map_weights(fc, true);
        }

        // Push particles in the velocity space
        // Use either MC or VR density estimtes in the rhs of the Poisson to get E
        if (rhsMode == RhsMode::VR)
            pc.kick_VR(fc);
        else
            pc.kick(fc);

        if (vr_enabled)
            pc.map_weights(fc, false);

        // MxE to conserve equil. moments.
        if (mxe_enabled)
            update_weights(pc, fc, sorter,
                           maximum_weight_recorder->device_max_mxe_iterations());
        
        // push particles in the position space
        pc.update_position();

        // update moments
        if (depositionMode == DepositionMode::SORTING)
            sorter.sort_particles_by_cell();

        compute_moments(pc, fc, sorter);

        if (vr_enabled) {
            // Record global importance weights after the complete time step.
            maximum_weight_recorder->record(step, step * DT, pc.d_w);
        }

        // print output
        if (field_output_enabled && step % 10 == 0)
            post_proc(fc, step);
    }

    std::cout << "Done.\n";
}
