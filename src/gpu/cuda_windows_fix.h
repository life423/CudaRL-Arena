#pragma once

// Fix for CUDA 12.9 + Windows SDK 10.0.26100.0 conflict
#ifdef _WIN32
    // Prevent Windows.h from including unnecessary headers
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    
    // CRITICAL: Include windows.h BEFORE any CUDA headers
    #include <windows.h>
    
    // Undefine problematic macros
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
    
    // Fix the _mm_popcnt_u64 conflict
    // This is defined in both intrin.h and CUDA headers
    #ifdef _MSC_VER
        #include <intrin.h>
        // Let CUDA's version take precedence
        #ifdef _mm_popcnt_u64
            #undef _mm_popcnt_u64
        #endif
    #endif
#endif
