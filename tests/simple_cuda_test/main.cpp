#include <iostream>
#include "cuda_runtime.h"
#include "kernels.cuh"

using namespace cudarl;

int main() {
    const int numElements = 10;
    size_t size = numElements * sizeof(float);
    
    // Allocate host memory
    float h_a[numElements], h_b[numElements], h_c[numElements];
    
    // Initialize host arrays
    for (int i = 0; i < numElements; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Copy host arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel using the wrapper function
    launchVectorAdd(d_a, d_b, d_c, numElements);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch vectorAdd kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < numElements; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}