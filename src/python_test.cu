#include <iostream>
#include <Python.h>
#include <cuda_runtime.h>

int main() {
    std::cout << "===== Python Integration Test =====" << std::endl;
    
    // Check CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices found: " << deviceCount << std::endl;
    
    // Initialize Python
    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Failed to initialize Python!" << std::endl;
        return 1;
    }
    
    // Print Python version
    std::cout << "Python version: " << Py_GetVersion() << std::endl;
    
    // Add current directory to Python path
    PyRun_SimpleString("import sys; sys.path.append('.')");
    PyRun_SimpleString("import sys; sys.path.append('./src')");
    
    // Try to import mvp.py
    int result = PyRun_SimpleString(
        "try:\n"
        "    import mvp\n"
        "    print('Successfully imported mvp module')\n"
        "    mvp.main()\n"
        "except ImportError as e:\n"
        "    print(f'Failed to import mvp: {e}')\n"
        "except Exception as e:\n"
        "    print(f'Error: {e}')\n"
    );
    
    if (result != 0) {
        std::cerr << "Error running Python code" << std::endl;
    }
    
    // Check for NumPy
    PyRun_SimpleString(
        "try:\n"
        "    import numpy\n"
        "    print('NumPy is available')\n"
        "except ImportError:\n"
        "    print('NumPy is not installed')\n"
    );
    
    // Finalize Python
    Py_Finalize();
    std::cout << "===== Test Complete =====" << std::endl;
    
    return 0;
}