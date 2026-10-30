#include "gaussian_kernels.h"

#include <algorithm>

#include "cuda/block_reduce.h"
#include "cuda/cuda_memory.h"
#include "rl/gaussian_math.h"

namespace cudarl::rl {

namespace {

constexpr int kThreads       = 256;
constexpr int kReduceThreads = 1024;

int blocks_for(int n) { return std::max(1, (n + kThreads - 1) / kThreads); }

__global__ void sample_kernel(const float* __restrict__ mean, const float* __restrict__ log_std, int n, int dim,
                              uint32_t seed, uint32_t stream_id, float* __restrict__ actions,
                              float* __restrict__ logp) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float* a = actions + row * dim;
    for (int j = 0; j < dim; ++j) {
        const uint32_t index = static_cast<uint32_t>(row * dim + j);
        a[j] = mean[row * dim + j] + expf(log_std[j]) * rng::normal(seed, stream_id, index);
    }
    logp[row] = gaussian_logp(mean + row * dim, log_std, a, dim);
}

__global__ void loss_grad_kernel(const float* __restrict__ mean, const float* __restrict__ log_std,
                                 const float* __restrict__ values, int n, int dim,
                                 const float* __restrict__ actions, const float* __restrict__ old_logp,
                                 const float* __restrict__ advantages, const float* __restrict__ returns,
                                 const float* __restrict__ adv_stats, PpoCoefficients coeffs,
                                 float* __restrict__ dmean, float* __restrict__ dlog_std_rows,
                                 float* __restrict__ dvalues, float* __restrict__ row_stats) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float adv = (advantages[i] - adv_stats[0]) / (adv_stats[1] + 1e-8f);
    const auto  t   = ppo_gaussian_row<float>(mean + i * dim, log_std, dim, actions + i * dim, old_logp[i], adv,
                                             values[i], returns[i], coeffs, 1.f / static_cast<float>(n),
                                             dmean + i * dim, dlog_std_rows + i * dim, dvalues + i);
    row_stats[i * kGaussianStats + 0] = t.pg_loss;
    row_stats[i * kGaussianStats + 1] = t.v_loss;
    row_stats[i * kGaussianStats + 2] = t.entropy;
    row_stats[i * kGaussianStats + 3] = t.approx_kl;
    row_stats[i * kGaussianStats + 4] = t.clipped;
}

__global__ void sum_columns_kernel(const float* __restrict__ x, int rows, int cols, float* __restrict__ out) {
    __shared__ float scratch[kReduceThreads];
    const int c = blockIdx.x;   // one block per column
    float s = 0.f;
    for (int i = threadIdx.x; i < rows; i += kReduceThreads) s += x[i * cols + c];
    const float total = block_reduce_sum<kReduceThreads>(s, scratch);
    if (threadIdx.x == 0) out[c] += total;
}

__global__ void gather_rows_kernel(Permutation perm, int first, int count, int width, const float* __restrict__ src,
                                   float* __restrict__ dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * width) return;
    const int      i   = idx / width;
    const int      col = idx % width;
    const uint32_t row = permute(perm, static_cast<uint32_t>(first + i));
    dst[idx]           = src[static_cast<int>(row) * width + col];
}

} // namespace

void sample_gaussian(const float* mean, const float* log_std, int n, int dim, uint32_t seed, uint32_t stream_id,
                     float* actions, float* logp, cudaStream_t stream) {
    sample_kernel<<<blocks_for(n), kThreads, 0, stream>>>(mean, log_std, n, dim, seed, stream_id, actions, logp);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void gaussian_ppo_loss_grad(const float* mean, const float* log_std, const float* values, int n, int dim,
                            const float* actions, const float* old_logp, const float* advantages,
                            const float* returns, const float* adv_stats, PpoCoefficients coeffs, float* dmean,
                            float* dlog_std_rows, float* dvalues, float* row_stats, float* stats, cudaStream_t stream) {
    loss_grad_kernel<<<blocks_for(n), kThreads, 0, stream>>>(mean, log_std, values, n, dim, actions, old_logp,
                                                             advantages, returns, adv_stats, coeffs, dmean,
                                                             dlog_std_rows, dvalues, row_stats);
    CUDA_CHECK_THROW(cudaGetLastError());
    sum_columns(row_stats, n, kGaussianStats, stats, stream);
}

void sum_columns(const float* x, int rows, int cols, float* out, cudaStream_t stream) {
    sum_columns_kernel<<<cols, kReduceThreads, 0, stream>>>(x, rows, cols, out);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void gather_rows(const Permutation& perm, int first, int count, int width, const float* src, float* dst,
                 cudaStream_t stream) {
    const int total = count * width;
    gather_rows_kernel<<<blocks_for(total), kThreads, 0, stream>>>(perm, first, count, width, src, dst);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::rl
