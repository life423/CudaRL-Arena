#include "loss.h"

#include <algorithm>

#include "cuda/cuda_memory.h"

namespace cudarl::nn {

namespace {

constexpr int kThreads = 256;

__global__ void mse_grad_kernel(const float* __restrict__ y, const float* __restrict__ t, float* __restrict__ dy,
                                int count) {
    const float scale = 2.f / static_cast<float>(count);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        dy[i] = scale * (y[i] - t[i]);
    }
}

} // namespace

void mse_loss_grad(const float* d_y, const float* d_target, float* d_dy, int count, cudaStream_t stream) {
    if (count <= 0) return;
    const int blocks = std::min((count + kThreads - 1) / kThreads, 4096);
    mse_grad_kernel<<<blocks, kThreads, 0, stream>>>(d_y, d_target, d_dy, count);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::nn
