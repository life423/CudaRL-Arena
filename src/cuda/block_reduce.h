#ifndef CUDARL_CUDA_BLOCK_REDUCE_H
#define CUDARL_CUDA_BLOCK_REDUCE_H

#include <cuda_runtime.h>

namespace cudarl {

// Tree reduction in shared memory over N threads: the same order every run, so
// results are deterministic. `scratch` needs N floats. Every thread gets the total.
template <int N>
__device__ float block_reduce_sum(float v, float* scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = N / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

} // namespace cudarl

#endif // CUDARL_CUDA_BLOCK_REDUCE_H
