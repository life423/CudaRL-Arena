#pragma once

#include "../core/environment.h"

namespace cudarl {

// CUDA kernel to reset the environment
__global__ void reset_environment(EnvironmentState* state);

// CUDA kernel to step the environment based on action
__global__ void step_environment(EnvironmentState* state, int action);

// Vector addition kernel (for testing)
template<typename T>
__global__ void vector_add(const T* A, const T* B, T* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

} // namespace cudarl