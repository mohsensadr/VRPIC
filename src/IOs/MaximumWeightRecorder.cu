#include "IOs/MaximumWeightRecorder.cuh"

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

}  // namespace

MaximumWeightRecorder::MaximumWeightRecorder(int particle_count,
                                             const std::string& filename)
    : particle_count_(particle_count) {
    if (particle_count_ <= 0) {
        throw std::invalid_argument("MaximumWeightRecorder requires at least one particle");
    }

    const std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_) {
        throw std::runtime_error("Could not open maximum-weight output file: " + filename);
    }
    output_ << "step,time,max_weight\n";
    output_ << std::setprecision(17);

    check_cuda(cudaMalloc(&device_maximum_, sizeof(float_type)),
               "allocating maximum-weight result");
    check_cuda(cub::DeviceReduce::Max(nullptr, temp_storage_bytes_,
                                     static_cast<const float_type*>(nullptr),
                                     device_maximum_, particle_count_),
               "sizing maximum-weight reduction workspace");
    check_cuda(cudaMalloc(&device_temp_storage_, temp_storage_bytes_),
               "allocating maximum-weight reduction workspace");
}

MaximumWeightRecorder::~MaximumWeightRecorder() {
    cudaFree(device_temp_storage_);
    cudaFree(device_maximum_);
}

void MaximumWeightRecorder::record(int step, float_type time,
                                   const float_type* device_weights) {
    check_cuda(cub::DeviceReduce::Max(device_temp_storage_, temp_storage_bytes_,
                                     device_weights, device_maximum_, particle_count_),
               "reducing maximum particle weight");

    float_type maximum;
    check_cuda(cudaMemcpy(&maximum, device_maximum_, sizeof(float_type),
                          cudaMemcpyDeviceToHost),
               "copying maximum particle weight");

    output_ << step << ',' << time << ',' << maximum << '\n';
    output_.flush();
    if (!output_) {
        throw std::runtime_error("Failed while writing maximum-weight diagnostic");
    }
}
