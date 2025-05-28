#pragma once

/**
 * RTX 5070 CUDA Optimization Configuration
 * 
 * Optimized settings for Ada Lovelace architecture
 * RTX 5070: 6,144 CUDA cores, 12GB GDDR7, SM 8.9
 */

#ifdef __CUDACC__
    #include <cuda_runtime.h>
#else
    // Forward declarations for non-CUDA compilers
    struct dim3 {
        unsigned int x, y, z;
        dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    };
    
    struct cudaDeviceProp {
        int major, minor;
        char name[256];
        size_t totalGlobalMem;
        int multiProcessorCount;
    };
    
    typedef int cudaError_t;
    #define cudaSuccess 0
    
    // Stub functions for non-CUDA compilation
    inline int cudaGetDevice(int* device) { *device = 0; return 0; }
    inline int cudaGetDeviceProperties(cudaDeviceProp* prop, int device) { 
        prop->major = 8; prop->minor = 9; return 0; 
    }
    inline int cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, 
                                                   void* func, size_t dynSharedMemPerBlock, int blockSizeLimit) {
        *minGridSize = 1; *blockSize = 256; return 0;
    }
#endif

#include <cstdio>

namespace cudarl {

// RTX 5070 Architecture Specifications
struct RTX5070Config {
    // Ada Lovelace SM 8.9 specifications
    static constexpr int COMPUTE_CAPABILITY_MAJOR = 8;
    static constexpr int COMPUTE_CAPABILITY_MINOR = 9;
    
    // RTX 5070 hardware specs
    static constexpr int CUDA_CORES = 6144;
    static constexpr int STREAMING_MULTIPROCESSORS = 48;  // Estimated for RTX 5070
    static constexpr int CORES_PER_SM = 128;
    static constexpr int MAX_THREADS_PER_SM = 2048;
    static constexpr int MAX_BLOCKS_PER_SM = 24;
    
    // Memory specifications
    static constexpr size_t GLOBAL_MEMORY_GB = 12;
    static constexpr size_t SHARED_MEMORY_PER_SM = 65536;  // 64KB
    static constexpr size_t L2_CACHE_SIZE = 64 * 1024 * 1024;  // 64MB estimated
    
    // Optimal kernel launch parameters
    static constexpr int OPTIMAL_BLOCK_SIZE_1D = 256;     // For 1D kernels
    static constexpr int OPTIMAL_BLOCK_SIZE_2D_X = 16;    // For 2D kernels
    static constexpr int OPTIMAL_BLOCK_SIZE_2D_Y = 16;
    
    // Memory coalescing parameters
    static constexpr int WARP_SIZE = 32;
    static constexpr int MEMORY_ALIGNMENT = 128;  // bytes
    
    // Performance optimization thresholds
    static constexpr int MIN_BLOCKS_FOR_OCCUPANCY = 192;  // 4 blocks per SM * 48 SMs
    static constexpr int PREFERRED_SHARED_MEMORY_USAGE = 32768;  // 32KB for good occupancy
};

// Utility functions for RTX 5070 optimization
class RTX5070Optimizer {
public:
    // Calculate optimal grid size for 1D kernels
    static inline dim3 calculateOptimalGrid1D(int totalElements, int blockSize = RTX5070Config::OPTIMAL_BLOCK_SIZE_1D) {
        int gridSize = (totalElements + blockSize - 1) / blockSize;
        return dim3(gridSize);
    }
    
    // Calculate optimal grid size for 2D kernels
    static inline dim3 calculateOptimalGrid2D(int width, int height, 
                                              int blockX = RTX5070Config::OPTIMAL_BLOCK_SIZE_2D_X,
                                              int blockY = RTX5070Config::OPTIMAL_BLOCK_SIZE_2D_Y) {
        int gridX = (width + blockX - 1) / blockX;
        int gridY = (height + blockY - 1) / blockY;
        return dim3(gridX, gridY);
    }
    
    // Check if current device is compatible
    static bool isRTX5070Compatible() {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        // Check for Ada Lovelace architecture (SM 8.9) or compatible
        return (props.major >= RTX5070Config::COMPUTE_CAPABILITY_MAJOR && 
                props.minor >= RTX5070Config::COMPUTE_CAPABILITY_MINOR);
    }
    
    // Get optimal occupancy for a kernel
    template<typename KernelFunc>
    static int getOptimalBlockSize(KernelFunc kernel) {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
        return blockSize;
    }
    
    // Memory bandwidth optimization check
    static bool isMemoryAccessCoalesced(size_t elementSize, size_t stride) {
        // Check if memory access pattern is optimal for RTX 5070
        return (stride * elementSize) % RTX5070Config::MEMORY_ALIGNMENT == 0;
    }
};

// Kernel launch macros optimized for RTX 5070 (only for CUDA compilation)
#ifdef __CUDACC__
    #define LAUNCH_RTX5070_KERNEL_1D(kernel, elements, ...) do { \
        dim3 blockSize(RTX5070Config::OPTIMAL_BLOCK_SIZE_1D); \
        dim3 gridSize = RTX5070Optimizer::calculateOptimalGrid1D(elements); \
        kernel<<<gridSize, blockSize>>>(__VA_ARGS__); \
    } while(0)

    #define LAUNCH_RTX5070_KERNEL_2D(kernel, width, height, ...) do { \
        dim3 blockSize(RTX5070Config::OPTIMAL_BLOCK_SIZE_2D_X, RTX5070Config::OPTIMAL_BLOCK_SIZE_2D_Y); \
        dim3 gridSize = RTX5070Optimizer::calculateOptimalGrid2D(width, height); \
        kernel<<<gridSize, blockSize>>>(__VA_ARGS__); \
    } while(0)
#else
    // Stub macros for non-CUDA compilation
    #define LAUNCH_RTX5070_KERNEL_1D(kernel, elements, ...)
    #define LAUNCH_RTX5070_KERNEL_2D(kernel, width, height, ...)
#endif

// Performance monitoring utilities
struct RTX5070PerfMonitor {
    static void printOptimalSettings() {
        printf("RTX 5070 Optimal Settings:\n");
        printf("  Block Size 1D: %d\n", RTX5070Config::OPTIMAL_BLOCK_SIZE_1D);
        printf("  Block Size 2D: %dx%d\n", RTX5070Config::OPTIMAL_BLOCK_SIZE_2D_X, RTX5070Config::OPTIMAL_BLOCK_SIZE_2D_Y);
        printf("  SMs: %d\n", RTX5070Config::STREAMING_MULTIPROCESSORS);
        printf("  CUDA Cores: %d\n", RTX5070Config::CUDA_CORES);
        printf("  Shared Memory per SM: %zu KB\n", RTX5070Config::SHARED_MEMORY_PER_SM / 1024);
    }
    
    static void checkDeviceCompatibility() {
        if (RTX5070Optimizer::isRTX5070Compatible()) {
            printf("✓ Device is RTX 5070 compatible (SM 8.9+)\n");
        } else {
            printf("⚠ Device may not be optimal for RTX 5070 settings\n");
        }
    }
};

} // namespace cudarl
