#ifndef CUDARL_RL_GAUSSIAN_KERNELS_H
#define CUDARL_RL_GAUSSIAN_KERNELS_H

#include <cstdint>

#include <cuda_runtime.h>

#include "rl/ppo_math.h"

// Batched kernels for the Gaussian policy. Like ppo_kernels, all are
// asynchronous on `stream`, deterministic (no atomics), and throw cuda_error on
// launch failure.
namespace cudarl::rl {

// Draws actions a = mean + exp(log_std) * eps for n rows of dim dimensions.
// Row i, dimension j uses noise from (seed, stream_id, i * dim + j).
// Also writes each row's log-probability of the drawn action.
void sample_gaussian(const float* mean, const float* log_std, int n, int dim, uint32_t seed, uint32_t stream_id,
                     float* actions, float* logp, cudaStream_t stream);

// PPO loss gradient for a minibatch of n samples. Advantages are normalized
// with adv_stats = (mean, std) first. Writes dmean [n x dim], dlog_std_rows
// [n x dim] (sum over rows for the shared log_std gradient), and dvalues [n].
// Adds this minibatch's summed (pg_loss, v_loss, entropy, approx_kl, clipped)
// to stats[0..kGaussianStats).
constexpr int kGaussianStats = 5;
void gaussian_ppo_loss_grad(const float* mean, const float* log_std, const float* values, int n, int dim,
                            const float* actions, const float* old_logp, const float* advantages,
                            const float* returns, const float* adv_stats, PpoCoefficients coeffs, float* dmean,
                            float* dlog_std_rows, float* dvalues, float* row_stats, float* stats, cudaStream_t stream);

// out[c] += sum over rows i of x[i * cols + c], for each column c.
void sum_columns(const float* x, int rows, int cols, float* out, cudaStream_t stream);

// Gathers rows of a [N x width] matrix in permuted order:
// dst[i * width + c] = src[permute(perm, first + i) * width + c] for i < count.
void gather_rows(const Permutation& perm, int first, int count, int width, const float* src, float* dst,
                 cudaStream_t stream);

// Episode totals over n environments, each entry of which is zero unless its episode ended:
// out[0] += sum of lengths, out[1] += sum of laps, out[2] += episodes ended,
// out[3] += episodes ended by a crash (done and not truncated).
void episode_totals(const float* lengths, const float* laps, const float* done, const float* truncated, int n,
                    float* out, cudaStream_t stream);

} // namespace cudarl::rl

#endif // CUDARL_RL_GAUSSIAN_KERNELS_H
