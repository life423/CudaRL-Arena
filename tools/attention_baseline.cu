#include "attention_baseline.h"

#include <cstddef>

#include "cuda/cuda_memory.h"

namespace {

constexpr int kWarpsPerBlock = 8;

// -infinity as a float. The INFINITY macro is a double constant, which nvcc rejects here.
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000u); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__global__ void softmax_rows_kernel(float* S, int rows, int cols, float scale) {
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    float* x = S + static_cast<size_t>(row) * cols;

    float mx = neg_inf();
    for (int c = lane; c < cols; c += 32) mx = fmaxf(mx, scale * x[c]);
    mx = warp_max(mx);

    float sum = 0.f;
    for (int c = lane; c < cols; c += 32) {
        const float e = expf(scale * x[c] - mx);
        x[c] = e;
        sum += e;
    }
    const float inv = 1.f / warp_sum(sum);
    for (int c = lane; c < cols; c += 32) x[c] *= inv;
}

__global__ void softmax_backward_rows_kernel(const float* P, float* dP, int rows, int cols, float scale) {
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* p = P + static_cast<size_t>(row) * cols;
    float* g = dP + static_cast<size_t>(row) * cols;

    float dot = 0.f;
    for (int c = lane; c < cols; c += 32) dot += p[c] * g[c];
    dot = warp_sum(dot);
    for (int c = lane; c < cols; c += 32) g[c] = scale * p[c] * (g[c] - dot);
}

} // namespace

void unfused_softmax_rows(float* S, int rows, int cols, float scale, cudaStream_t stream) {
    const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    softmax_rows_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(S, rows, cols, scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void unfused_softmax_backward_rows(const float* P, float* dP, int rows, int cols, float scale,
                                   cudaStream_t stream) {
    const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    softmax_backward_rows_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(P, dP, rows, cols, scale);
    CUDA_CHECK_THROW(cudaGetLastError());
}
