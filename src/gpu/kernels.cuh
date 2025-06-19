#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <vector>

namespace cudarl {
    // Simple C++ function that handles everything
    std::vector<float> addVectors(const std::vector<float>& a, const std::vector<float>& b);
}

#endif // KERNELS_CUH