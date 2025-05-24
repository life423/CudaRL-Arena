#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cudarl {

// CUDA error checking helper
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + file + " at line " + 
            std::to_string(line) + ": " + cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

// RAII wrapper for CUDA memory
template<typename T>
class CudaMemory {
private:
    T* d_ptr = nullptr;
    size_t elements = 0;

public:
    CudaMemory() = default;
    
    explicit CudaMemory(size_t count) : elements(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
        }
    }
    
    ~CudaMemory() {
        free();
    }
    
    // Move semantics
    CudaMemory(CudaMemory&& other) noexcept : d_ptr(other.d_ptr), elements(other.elements) {
        other.d_ptr = nullptr;
        other.elements = 0;
    }
    
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            free();
            d_ptr = other.d_ptr;
            elements = other.elements;
            other.d_ptr = nullptr;
            other.elements = 0;
        }
        return *this;
    }
    
    // Disable copy
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
    // Allocate memory
    void allocate(size_t count) {
        free();
        if (count > 0) {
            elements = count;
            CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
        }
    }
    
    // Free memory
    void free() {
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
        }
        elements = 0;
    }
    
    // Copy from host to device
    void copyFromHost(const T* h_ptr, size_t count) {
        if (count > elements || !d_ptr) {
            throw std::runtime_error("Invalid CUDA memory operation in copyFromHost");
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // Copy from device to host
    void copyToHost(T* h_ptr, size_t count) const {
        if (count > elements || !d_ptr) {
            throw std::runtime_error("Invalid CUDA memory operation in copyToHost");
        }
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    // Get device pointer
    T* get() const { return d_ptr; }
    
    // Get element count
    size_t size() const { return elements; }
    
    // Check if allocated
    bool isAllocated() const { return d_ptr != nullptr; }
    
    // Cast operator
    operator T*() const { return d_ptr; }
};

// Device information utility
struct CudaDeviceInfo {
    static void printDeviceInfo() {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found!");
        }
        
        std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
        
        // Get properties for each device
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp deviceProp;
            CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));
            
            std::cout << "\nDevice " << i << ": " << deviceProp.name << std::endl;
            std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
            std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        }
    }
};

} // namespace cudarl