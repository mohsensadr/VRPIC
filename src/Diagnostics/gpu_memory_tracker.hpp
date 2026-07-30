#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// Track device allocations owned directly by VRPIC. CUDA context, cuFFT
// internal, and allocations belonging to other processes are intentionally not
// included, making the reported value reproducible across otherwise identical
// runs.
void reset_gpu_memory_tracker();
std::size_t peak_gpu_memory_bytes();

cudaError_t tracked_cuda_malloc_impl(void** pointer, std::size_t bytes);
cudaError_t tracked_cuda_free(void* pointer);

template <typename T>
cudaError_t tracked_cuda_malloc(T** pointer, std::size_t bytes) {
    return tracked_cuda_malloc_impl(reinterpret_cast<void**>(pointer), bytes);
}
