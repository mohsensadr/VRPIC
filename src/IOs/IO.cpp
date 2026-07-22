// IO.cpp
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#include "IOs/IO.h"
#include "Constants/constants.hpp"
#include <cuda_runtime.h>

namespace fs = std::filesystem;

void write_to_csv(const std::string& filename, float_type *x) {
    std::ofstream out(filename);
    for (int j = 0; j < N_GRID_Y; ++j) {
        for (int i = 0; i < N_GRID_X; ++i) {
            out << x[i + j * N_GRID_X];
            if (i < N_GRID_X - 1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

void write_output(int step, float_type* x, std::string s) {
    std::ostringstream oss;
    fs::create_directories("data");
    oss << "data/" << s << "_step_" << std::setw(3) << std::setfill('0') << step << ".csv";
    write_to_csv(oss.str(), x);
}

void post_proc(FieldContainer &fc, int step){

    std::vector<float_type> h_var(grid_size);

    auto dump = [&](float_type* device_ptr, const std::string& label) {
        cudaError_t err = cudaMemcpy(h_var.data(), device_ptr, sizeof(float_type) * grid_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
            return;
        } 
        write_output(step, h_var.data(), label);
    };

    dump(fc.d_N, "N");
    dump(fc.d_Ux, "Ux");
    dump(fc.d_Uy, "Uy");
    dump(fc.d_T, "T");
    dump(fc.d_NVR, "NVR");
    dump(fc.d_UxVR, "UxVR");
    dump(fc.d_UyVR, "UyVR");
    dump(fc.d_TVR, "TVR");
    dump(fc.d_phi, "phi");
    dump(fc.d_Ex, "Ex");
    dump(fc.d_Ey, "Ey");
    dump(fc.d_phiVR, "phiVR");
    dump(fc.d_ExVR, "ExVR");
    dump(fc.d_EyVR, "EyVR");

    std::cout << "Wrote postproc in step: " << step << std::endl;
    cudaDeviceSynchronize();
}

void write_performance_metrics(double execution_time_seconds,
                               std::size_t peak_device_memory_bytes) {
    fs::create_directories("data");
    const fs::path filename = fs::path("data") / "performance_metrics.csv";
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("Could not open performance metrics file: " +
                                 filename.string());
    }

    const double memory_mb =
        static_cast<double>(peak_device_memory_bytes) / 1'000'000.0;
    const double particles_per_cell =
        static_cast<double>(N_PARTICLES) / static_cast<double>(grid_size);

    output << "particles_per_cell,total_particles,execution_time_s,memory_mb\n";
    output << std::setprecision(17) << particles_per_cell << ',' << N_PARTICLES
           << ',' << execution_time_seconds << ',' << memory_mb << '\n';
    if (!output) {
        throw std::runtime_error("Failed while writing performance metrics: " +
                                 filename.string());
    }
}
