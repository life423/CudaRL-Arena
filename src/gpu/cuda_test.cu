#include <iostream>
#include "../core/cuda_utils.h"
#include "kernels.cuh"

using namespace cudarl;

// Function to test basic CUDA computation
bool testCudaComputation() {
    const int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    // Allocate host memory using std::vector for RAII
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);
    
    // Initialize host arrays
    std::mt19937 rng(std::time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < numElements; i++) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }
    
    try {
        // Allocate device memory using RAII wrapper
        CudaMemory<float> d_A(numElements);
        CudaMemory<float> d_B(numElements);
        CudaMemory<float> d_C(numElements);
        
        // Copy data from host to device
        d_A.copyFromHost(h_A.data(), numElements);
        d_B.copyFromHost(h_B.data(), numElements);
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        
        std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
        
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), numElements);
        CUDA_CHECK(cudaGetLastError());
        
        // Wait for GPU to finish
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back to host
        d_C.copyToHost(h_C.data(), numElements);
        
        // Verify result
        bool success = true;
        for (int i = 0; i < numElements; i++) {
            float expected = h_A[i] + h_B[i];
            if (fabs(h_C[i] - expected) > 1e-5) {
                std::cerr << "Verification failed at element " << i << ": " 
                          << h_C[i] << " != " << expected << std::endl;
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "Vector addition test PASSED!" << std::endl;
        } else {
            std::cout << "Vector addition test FAILED!" << std::endl;
        }
        
        return success;
    } catch (const std::exception& e) {
        std::cerr << "CUDA test error: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "===== CUDA Functionality Test =====" << std::endl;
    
    try {
        // Check CUDA device properties
        CudaDeviceInfo::printDeviceInfo();
        
        // Test basic CUDA computation
        bool computationSuccess = testCudaComputation();
        
        if (computationSuccess) {
            std::cout << "\n✓ CUDA core functionality is working correctly!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ CUDA core functionality test failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}