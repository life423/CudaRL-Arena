#include "ppo_kernels.h"

#include <algorithm>

#include "cuda/cuda_memory.h"

namespace cudarl::rl {

namespace {

constexpr int kThreads       = 256;
constexpr int kReduceThreads = 1024;

int blocks_for(int n) { return std::max(1, (n + kThreads - 1) / kThreads); }

// Tree reduction in shared memory: the same order every run.
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

__global__ void sample_kernel(const float* __restrict__ logits, int n, int k, uint32_t seed, uint32_t stream_id,
                              bool greedy, int* __restrict__ actions, float* __restrict__ logp_out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float logp[kMaxActions];
    log_softmax(logits + row * k, k, logp);
    const float u = rng::uniform(seed, stream_id, static_cast<uint32_t>(row));
    const int a = sample_action(logp, k, u, greedy);
    actions[row]  = a;
    logp_out[row] = logp[a];
}

__global__ void gae_kernel(int steps, int num_envs, const float* __restrict__ rewards, const float* __restrict__ values,
                           const float* __restrict__ dones, const float* __restrict__ truncated,
                           const float* __restrict__ final_values, const float* __restrict__ last_values, float gamma,
                           float lambda, float* __restrict__ advantages, float* __restrict__ returns) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;
    gae_one_env(steps, num_envs, env, rewards, values, dones, truncated, final_values, last_values, gamma, lambda,
                advantages, returns);
}

__global__ void gather_kernel(Permutation perm, int first, int count, int obs_dim, const float* __restrict__ obs,
                              const int* __restrict__ actions, const float* __restrict__ logp,
                              const float* __restrict__ advantages, const float* __restrict__ returns,
                              float* __restrict__ mb_obs, int* __restrict__ mb_actions, float* __restrict__ mb_logp,
                              float* __restrict__ mb_advantages, float* __restrict__ mb_returns) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t src = permute(perm, static_cast<uint32_t>(first + i));
    for (int d = 0; d < obs_dim; ++d) mb_obs[i * obs_dim + d] = obs[src * obs_dim + d];
    mb_actions[i]    = actions[src];
    mb_logp[i]       = logp[src];
    mb_advantages[i] = advantages[src];
    mb_returns[i]    = returns[src];
}

__global__ void mean_std_kernel(const float* __restrict__ x, int n, float* __restrict__ out) {
    __shared__ float scratch[kReduceThreads];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += kReduceThreads) s += x[i];
    const float mean = block_reduce_sum<kReduceThreads>(s, scratch) / static_cast<float>(n);
    float sq = 0.f;
    for (int i = threadIdx.x; i < n; i += kReduceThreads) sq += (x[i] - mean) * (x[i] - mean);
    const float var = block_reduce_sum<kReduceThreads>(sq, scratch) / static_cast<float>(n);
    if (threadIdx.x == 0) {
        out[0] = mean;
        out[1] = sqrtf(var);
    }
}

__global__ void ppo_grad_kernel(const float* __restrict__ logits, const float* __restrict__ values, int n, int k,
                                const int* __restrict__ actions, const float* __restrict__ old_logp,
                                const float* __restrict__ advantages, const float* __restrict__ returns,
                                const float* __restrict__ adv_stats, PpoCoefficients coeffs,
                                float* __restrict__ dlogits, float* __restrict__ dvalues,
                                float* __restrict__ row_stats) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float adv = (advantages[i] - adv_stats[0]) / (adv_stats[1] + 1e-8f);
    const auto t = ppo_row<float>(logits + i * k, k, actions[i], old_logp[i], adv, values[i], returns[i], coeffs,
                                  1.f / static_cast<float>(n), dlogits + i * k, dvalues + i);
    row_stats[0 * n + i] = t.pg_loss;
    row_stats[1 * n + i] = t.v_loss;
    row_stats[2 * n + i] = t.entropy;
    row_stats[3 * n + i] = t.approx_kl;
    row_stats[4 * n + i] = t.clipped;
}

// Sums each of `rows` stat columns of length n and adds the totals to out[0..rows).
__global__ void sum_columns_kernel(const float* __restrict__ x, int n, int rows, float* __restrict__ out) {
    __shared__ float scratch[kReduceThreads];
    for (int r = 0; r < rows; ++r) {
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += kReduceThreads) s += x[r * n + i];
        const float total = block_reduce_sum<kReduceThreads>(s, scratch);
        if (threadIdx.x == 0) out[r] += total;
    }
}

__global__ void episode_stats_kernel(const float* __restrict__ lengths, int n, float* __restrict__ out) {
    __shared__ float scratch[kReduceThreads];
    float sum = 0.f, count = 0.f;
    for (int i = threadIdx.x; i < n; i += kReduceThreads) {
        sum += lengths[i];
        count += lengths[i] > 0.f ? 1.f : 0.f;
    }
    const float total_sum   = block_reduce_sum<kReduceThreads>(sum, scratch);
    const float total_count = block_reduce_sum<kReduceThreads>(count, scratch);
    if (threadIdx.x == 0) {
        out[0] += total_sum;
        out[1] += total_count;
    }
}

} // namespace

void sample_actions(const float* logits, int n, int k, uint32_t seed, uint32_t stream_id, bool greedy, int* actions,
                    float* logp, cudaStream_t stream) {
    sample_kernel<<<blocks_for(n), kThreads, 0, stream>>>(logits, n, k, seed, stream_id, greedy, actions, logp);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void compute_gae(int steps, int num_envs, const float* rewards, const float* values, const float* dones,
                 const float* truncated, const float* final_values, const float* last_values, float gamma,
                 float lambda, float* advantages, float* returns, cudaStream_t stream) {
    gae_kernel<<<blocks_for(num_envs), kThreads, 0, stream>>>(steps, num_envs, rewards, values, dones, truncated,
                                                              final_values, last_values, gamma, lambda, advantages,
                                                              returns);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void gather_minibatch(const Permutation& perm, int first, int count, int obs_dim, const float* obs, const int* actions,
                      const float* logp, const float* advantages, const float* returns, float* mb_obs, int* mb_actions,
                      float* mb_logp, float* mb_advantages, float* mb_returns, cudaStream_t stream) {
    gather_kernel<<<blocks_for(count), kThreads, 0, stream>>>(perm, first, count, obs_dim, obs, actions, logp,
                                                              advantages, returns, mb_obs, mb_actions, mb_logp,
                                                              mb_advantages, mb_returns);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void mean_std(const float* x, int n, float* out, cudaStream_t stream) {
    mean_std_kernel<<<1, kReduceThreads, 0, stream>>>(x, n, out);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void ppo_loss_grad(const float* logits, const float* values, int n, int k, const int* actions, const float* old_logp,
                   const float* advantages, const float* returns, const float* adv_stats, PpoCoefficients coeffs,
                   float* dlogits, float* dvalues, float* row_stats, float* stats, cudaStream_t stream) {
    ppo_grad_kernel<<<blocks_for(n), kThreads, 0, stream>>>(logits, values, n, k, actions, old_logp, advantages,
                                                            returns, adv_stats, coeffs, dlogits, dvalues, row_stats);
    sum_columns_kernel<<<1, kReduceThreads, 0, stream>>>(row_stats, n, kPpoStats, stats);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void episode_stats(const float* finished_length, int n, float* out, cudaStream_t stream) {
    episode_stats_kernel<<<1, kReduceThreads, 0, stream>>>(finished_length, n, out);
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace cudarl::rl
