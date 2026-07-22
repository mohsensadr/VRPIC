// field_container.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "Constants/constants.hpp"
#include "Diagnostics/gpu_memory_tracker.hpp"

#define TILE_X 16
#define TILE_Y 16

class FieldContainer {
public:
    float_type *d_N = nullptr;
    float_type *d_Ux = nullptr;
    float_type *d_Uy = nullptr;
    float_type *d_T = nullptr;
    float_type *d_phi = nullptr;
    float_type *d_Ex = nullptr;
    float_type *d_Ey = nullptr;

    float_type *d_NVR = nullptr;
    float_type *d_UxVR = nullptr;
    float_type *d_UyVR = nullptr;
    float_type *d_TVR = nullptr;
    float_type *d_phiVR = nullptr;
    float_type *d_ExVR = nullptr;
    float_type *d_EyVR = nullptr;

    float_type *d_pt0 = nullptr;
    float_type *d_pt1 = nullptr;
    float_type *d_pt2 = nullptr;

    float_type dx, dy;
    float_type xmin, ymin;
    int nx, ny;
    size_t grid_size;

    FieldContainer(int N_GRID_X, int N_GRID_Y, float_type Lx, float_type Ly) : nx(N_GRID_X), ny(N_GRID_Y) {
        grid_size = nx * ny;
        xmin = 0.0;
        ymin = 0.0;
        dx = Lx / nx;
        dy = Ly / ny;
        size_t bytes = grid_size * sizeof(float_type);

        tracked_cuda_malloc(&d_N, bytes);
        tracked_cuda_malloc(&d_Ux, bytes);
        tracked_cuda_malloc(&d_Uy, bytes);
        tracked_cuda_malloc(&d_T, bytes);
        tracked_cuda_malloc(&d_phi, bytes);
        tracked_cuda_malloc(&d_Ex, bytes);
        tracked_cuda_malloc(&d_Ey, bytes);

        tracked_cuda_malloc(&d_NVR, bytes);
        tracked_cuda_malloc(&d_UxVR, bytes);
        tracked_cuda_malloc(&d_UyVR, bytes);
        tracked_cuda_malloc(&d_TVR, bytes);
        tracked_cuda_malloc(&d_phiVR, bytes);
        tracked_cuda_malloc(&d_ExVR, bytes);
        tracked_cuda_malloc(&d_EyVR, bytes);

        tracked_cuda_malloc(&d_pt0, bytes);
        tracked_cuda_malloc(&d_pt1, bytes);
        tracked_cuda_malloc(&d_pt2, bytes);
    }

    ~FieldContainer() {
        tracked_cuda_free(d_N);
        tracked_cuda_free(d_Ux);
        tracked_cuda_free(d_Uy);
        tracked_cuda_free(d_T);
        tracked_cuda_free(d_phi);
        tracked_cuda_free(d_Ex);
        tracked_cuda_free(d_Ey);

        tracked_cuda_free(d_NVR);
        tracked_cuda_free(d_UxVR);
        tracked_cuda_free(d_UyVR);
        tracked_cuda_free(d_TVR);
        tracked_cuda_free(d_phiVR);
        tracked_cuda_free(d_ExVR);
        tracked_cuda_free(d_EyVR);

        tracked_cuda_free(d_pt0);
        tracked_cuda_free(d_pt1);
        tracked_cuda_free(d_pt2);
    }

    // Optional: zero out all field arrays
    void setZero() {
        size_t bytes = grid_size * sizeof(float_type);
        cudaMemset(d_N, 0, bytes);
        cudaMemset(d_Ux, 0, bytes);
        cudaMemset(d_Uy, 0, bytes);
        cudaMemset(d_T, 0, bytes);

        cudaMemset(d_NVR, 0, bytes);
        cudaMemset(d_UxVR, 0, bytes);
        cudaMemset(d_UyVR, 0, bytes);
        cudaMemset(d_TVR, 0, bytes);
      
        cudaMemset(d_pt0, 0, bytes);
        cudaMemset(d_pt1, 0, bytes);
        cudaMemset(d_pt2, 0, bytes);
    }
};
