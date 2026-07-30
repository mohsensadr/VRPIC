#include "Diagnostics/gpu_memory_tracker.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace {
std::mutex tracker_mutex;
std::unordered_map<void*, std::size_t> allocation_sizes;
std::size_t current_bytes = 0;
std::size_t maximum_bytes = 0;
}  // namespace

void reset_gpu_memory_tracker() {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    allocation_sizes.clear();
    current_bytes = 0;
    maximum_bytes = 0;
}

std::size_t peak_gpu_memory_bytes() {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    return maximum_bytes;
}

cudaError_t tracked_cuda_malloc_impl(void** pointer, std::size_t bytes) {
    const cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess) {
        return status;
    }

    std::lock_guard<std::mutex> lock(tracker_mutex);
    allocation_sizes[*pointer] = bytes;
    current_bytes += bytes;
    maximum_bytes = std::max(maximum_bytes, current_bytes);
    return status;
}

cudaError_t tracked_cuda_free(void* pointer) {
    const cudaError_t status = cudaFree(pointer);
    if (status != cudaSuccess || pointer == nullptr) {
        return status;
    }

    std::lock_guard<std::mutex> lock(tracker_mutex);
    const auto allocation = allocation_sizes.find(pointer);
    if (allocation != allocation_sizes.end()) {
        current_bytes -= allocation->second;
        allocation_sizes.erase(allocation);
    }
    return status;
}
