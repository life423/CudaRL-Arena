#include <iostream>
#include <cuda_runtime.h>
#include <random>

// Simple CUDA kernel to add two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Function to check CUDA device properties
void checkCudaDevice() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get properties for each device
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "\nDevice " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  CUDA cores: " << deviceProp.multiProcessorCount * 
                    (deviceProp.major >= 12 ? 128 :
                     deviceProp.major >= 9 ? 128 :
                     deviceProp.major == 8 ? 128 : 
                     deviceProp.major == 7 ? 64 : 
                     deviceProp.major <= 6 ? 128 : 0) << std::endl;
        std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    }
}

// Function to test basic CUDA computation
bool testCudaComputation() {
    const int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[numElements];
    float *h_B = new float[numElements];
    float *h_C = new float[numElements];
    
    // Initialize host arrays
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    
    cudaError_t error = cudaMalloc((void**)&d_A, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for A: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc((void**)&d_B, size);
    if (error != cudaSuccess) {
        cudaFree(d_A);
        std::cerr << "Failed to allocate device memory for B: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc((void**)&d_C, size);
    if (error != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        std::cerr << "Failed to allocate device memory for C: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy data from host to device
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data from host to device for A: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data from host to device for B: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Failed to launch kernel: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy result from device to host: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
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
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return success;
}

int main() {
    std::cout << "===== CUDA Functionality Test =====" << std::endl;
    
    // Check CUDA device properties
    checkCudaDevice();
    
    // Test basic CUDA computation
    bool computationSuccess = testCudaComputation();
    
    if (computationSuccess) {
        std::cout << "\n✓ CUDA core functionality is working correctly!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ CUDA core functionality test failed!" << std::endl;
        return 1;
    }
}