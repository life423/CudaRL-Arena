#ifndef KERNELS_CUH
#define KERNELS_CUH

namespace cudarl {
    // Wrapper function that can be called from regular C++ code
    void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int numElements);
}

#endif // KERNELS_CUH