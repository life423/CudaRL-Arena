#include <iostream>
#include "environment.h"

// Macro for CUDA error checking (retained for future use)
#define CHECK_CUDA(call) do { \
    cudaError_t _err = call; \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error %d: %s\n", _err, cudaGetErrorString(_err)); \
        throw std::runtime_error("CUDA call failed"); \
    } \
} while(0)

/*
 * Main entry point for the CUDA RL Arena PoC.
 * Demonstrates modular usage of the Environment class.
 */
int main() {
    std::cout << "Creating environment..." << std::endl;
    Environment env(0);

    std::cout << "Resetting environment (should launch GPU kernel)..." << std::endl;
    env.reset();

    std::cout << "Stepping environment with action=1..." << std::endl;
    env.step(1);

    std::cout << "Back on the CPU, PoC complete!" << std::endl;

    // you can invoke Python here via the C API, e.g.:
    // Py_Initialize();
    // PyRun_SimpleString("import mvp; mvp.main()");
    // Py_Finalize();

    return 0;
}
