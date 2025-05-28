#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, error, \
                cudaGetErrorName(error), cudaGetErrorString(error)); \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
    } \
} while(0)

// Safe division macro
#define DIVUP(n, d) (((n) + (d) - 1) / (d))

// Memory management template for RAII
template<typename T>
class DeviceBuffer {
private:
    T* ptr = nullptr;
    size_t size = 0;
    
public:
    explicit DeviceBuffer(size_t n) : size(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }
    
    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    
    // Delete copy constructor and assignment
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }
    
    // Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    T* get() { return ptr; }
    const T* get() const { return ptr; }
    size_t length() const { return size; }
    bool valid() const { return ptr != nullptr; }
};

// GPU memory info utilities
struct GpuMemoryInfo {
    size_t free;
    size_t total;
    size_t used;
    
    void update() {
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        used = total - free;
    }
    
    void print() const {
        printf("GPU Memory: %.1f MB used / %.1f MB total (%.1f%% used)\n",
               used / 1048576.0, total / 1048576.0, 
               100.0 * used / total);
    }
};

// Device info functions
inline void checkCudaDevice() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found!");
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
}

// Timing utilities
class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        CUDA_CHECK(cudaEventRecord(start));
    }
    
    float endTimer() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};

// Optimal launch parameters for RTX 5070
inline void getOptimalLaunchParams(int totalThreads, int& blockSize, int& gridSize) {
    // RTX 5070 specific optimizations
    blockSize = 256;  // Good balance for Ada Lovelace
    gridSize = DIVUP(totalThreads, blockSize);
    
    // Don't exceed SM capacity (48 SMs on RTX 5070)
    const int maxGridSize = 48 * 8;  // ~8 blocks per SM
    if (gridSize > maxGridSize) {
        gridSize = maxGridSize;
    }
}
