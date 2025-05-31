#include "kernels.cuh"
#include "cuda_runtime.h"

namespace cudarl {
    // The actual kernel
    __global__ void vectorAdd(const float* a, const float* b, float* c, int numElements) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < numElements) {
            c[idx] = a[idx] + b[idx];
        }
    }
    
    // Wrapper function that launches the kernel
    void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int numElements) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    }
}