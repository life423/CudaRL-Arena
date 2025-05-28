#pragma once

/**
 * CUDA Windows SDK Compatibility Header
 * 
 * Fixes CUDA 12.9 + Windows SDK 10.0.26100.0 conflicts
 * Automatically included for all CUDA files via CMake when needed
 */

#ifdef _WIN32

// Prevent Windows.h from including unnecessary headers that conflict with CUDA
#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
    #define NOMINMAX
#endif

// Include Windows headers BEFORE any CUDA headers to prevent conflicts
#include <windows.h>

// Undefine problematic macros that conflict with CUDA
#ifdef min
    #undef min
#endif

#ifdef max
    #undef max
#endif

// Fix the critical _mm_popcnt_u64 conflict between Windows SDK and CUDA
#ifdef _MSC_VER
    #include <intrin.h>
    
    // Let CUDA's version take precedence over Windows SDK version
    #ifdef _mm_popcnt_u64
        #undef _mm_popcnt_u64
    #endif
    
    // Additional Windows SDK conflicts
    #ifdef _mm_popcnt_u32
        #undef _mm_popcnt_u32
    #endif
#endif

// Prevent additional Windows SDK conflicts
#ifdef ERROR
    #undef ERROR
#endif

#ifdef IGNORE
    #undef IGNORE
#endif

// Define compatibility macros for common operations
#ifndef CUDA_WINDOWS_COMPAT_DEFINED
    #define CUDA_WINDOWS_COMPAT_DEFINED
    
    // Provide safe min/max implementations if needed
    #ifdef __CUDACC__
        // CUDA compiler version with device/host decorators
        template<typename T>
        __host__ __device__ inline T cuda_min(T a, T b) {
            return (a < b) ? a : b;
        }
        
        template<typename T>
        __host__ __device__ inline T cuda_max(T a, T b) {
            return (a > b) ? a : b;
        }
    #else
        // Regular C++ compiler version
        template<typename T>
        inline T cuda_min(T a, T b) {
            return (a < b) ? a : b;
        }
        
        template<typename T>
        inline T cuda_max(T a, T b) {
            return (a > b) ? a : b;
        }
    #endif
#endif

#endif // _WIN32
