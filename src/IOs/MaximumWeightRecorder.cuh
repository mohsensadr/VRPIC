#pragma once

#include "Constants/constants.hpp"

#include <cstddef>
#include <fstream>
#include <string>

/**
 * Record maximum particle-weight and MxE-iteration diagnostics each time step.
 *
 * The reduction is performed on the GPU.  Only the resulting scalar is copied
 * to the host, which keeps the diagnostic inexpensive for large particle sets.
 */
class MaximumWeightRecorder {
public:
    MaximumWeightRecorder(int particle_count,
                          const std::string& filename = "data/max_weight.csv");
    ~MaximumWeightRecorder();

    MaximumWeightRecorder(const MaximumWeightRecorder&) = delete;
    MaximumWeightRecorder& operator=(const MaximumWeightRecorder&) = delete;

    void begin_step();
    int* device_max_mxe_iterations() { return device_max_mxe_iterations_; }
    void record(int step, float_type time, const float_type* device_weights);

private:
    int particle_count_;
    void* device_temp_storage_ = nullptr;
    std::size_t temp_storage_bytes_ = 0;
    float_type* device_maximum_ = nullptr;
    int* device_max_mxe_iterations_ = nullptr;
    std::ofstream output_;
};
