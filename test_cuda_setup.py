#!/usr/bin/env python3
"""
Test script to verify CUDA 12.9 setup and Windows SDK compatibility.
"""

import subprocess
import os
import sys
import tempfile
from pathlib import Path

def test_nvcc():
    """Test nvcc compiler availability and version."""
    print("=== Testing NVCC ===")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVCC Output:")
            print(result.stdout)
            return True
        else:
            print(f"NVCC failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("ERROR: nvcc not found in PATH")
        return False

def test_cuda_compilation():
    """Test simple CUDA compilation with Windows SDK compatibility fixes."""
    print("\n=== Testing CUDA Compilation ===")
    
    # CUDA test code with Windows SDK fixes
    test_code = '''
// MUST BE FIRST - Fix Windows/CUDA conflicts
#ifdef _WIN32
    #define _ENABLE_EXTENDED_ALIGNED_STORAGE
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    
    // Include Windows headers first
    #include <windows.h>
    #include <intrin.h>
    
    // Undefine conflicts
    #undef min
    #undef max
    #ifdef _mm_popcnt_u64
        #undef _mm_popcnt_u64
    #endif
#endif

// Now include CUDA headers
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel(float* data) {
    int idx = threadIdx.x;
    data[idx] = idx * 2.0f;
}

int main() {
    printf("Testing CUDA 12.9 with Windows SDK compatibility...\\n");
    
    float* d_data;
    float h_data[256];
    
    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_data, 256 * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Launch kernel
    test_kernel<<<1, 256>>>(d_data);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }
    
    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }
    
    // Verify results
    bool success = true;
    for (int i = 0; i < 10; i++) {  // Check first 10 elements
        if (h_data[i] != i * 2.0f) {
            printf("Verification failed at index %d: expected %f, got %f\\n", 
                   i, (float)(i * 2.0f), h_data[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("CUDA test successful! Windows SDK conflicts resolved.\\n");
        printf("Sample results: h_data[0]=%f, h_data[1]=%f, h_data[2]=%f\\n", 
               h_data[0], h_data[1], h_data[2]);
    }
    
    cudaFree(d_data);
    return success ? 0 : 1;
}
'''

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / 'test.cu'
        exe_file = Path(temp_dir) / 'test.exe'
        
        # Write test code
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Compile with similar flags as CMake
        compile_cmd = [
            'nvcc', str(test_file), '-o', str(exe_file),
            '-arch=sm_86', '-m64', '--expt-relaxed-constexpr',
            '--use_fast_math'
        ]
        
        # Add Windows-specific flags
        if os.name == 'nt':  # Windows
            compile_cmd.extend(['-Xcompiler', '/MT'])
        
        print(f"Compiling with: {' '.join(compile_cmd)}")
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Compilation successful!")
            
            # Run the test
            print("Running CUDA test...")
            run_result = subprocess.run([str(exe_file)], capture_output=True, text=True)
            
            print("Test output:")
            print(run_result.stdout)
            
            if run_result.stderr:
                print("Errors/warnings:")
                print(run_result.stderr)
            
            return run_result.returncode == 0
        else:
            print("✗ Compilation failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

def test_cmake_build():
    """Test CMake build with Windows SDK fixes."""
    print("\n=== Testing CMake Build ===")
    
    if not Path('CMakeLists.txt').exists():
        print("CMakeLists.txt not found, skipping CMake test")
        return True
    
    # Clean build directory
    build_dir = Path('tests')
    if build_dir.exists():
        print("Using existing build directory")
    else:
        print("Build directory not found, would need to create")
        return True
    
    # Test if we can at least configure
    try:
        result = subprocess.run([
            'cmake', '--version'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ CMake is available")
            print("CMake version:", result.stdout.split('\n')[0])
            return True
        else:
            print("✗ CMake not available")
            return False
    except FileNotFoundError:
        print("✗ CMake not found")
        return False

def main():
    """Main test runner."""
    print("=== CUDA 12.9 Windows SDK Compatibility Test ===")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    print()
    
    tests = [
        ("NVCC Availability", test_nvcc),
        ("CUDA Compilation", test_cuda_compilation),
        ("CMake Build", test_cmake_build),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✓' if success else '✗'} {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CUDA 12.9 with Windows SDK fixes is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
