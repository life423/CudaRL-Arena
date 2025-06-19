#include "kernels.cuh"
#include "cuda_runtime.h"
#include <iostream>

namespace cudarl {
    // The GPU kernel
    __global__ void vectorAdd(const float* a, const float* b, float* c, int numElements) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < numElements) {
            c[idx] = a[idx] + b[idx];
        }
    }
    
    // All CUDA operations wrapped in one function
    std::vector<float> addVectors(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vectors must be same size");
        }
        
        int numElements = a.size();
        size_t size = numElements * sizeof(float);
        std::vector<float> result(numElements);
        
        // Allocate device memory
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);
        
        // Copy to device
        cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
        }
        
        // Copy result back
        cudaMemcpy(result.data(), d_c, size, cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        return result;
    }
}