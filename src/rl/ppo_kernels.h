#ifndef CUDARL_RL_PPO_KERNELS_H
#define CUDARL_RL_PPO_KERNELS_H

#include <cstdint>

#include <cuda_runtime.h>

#include "ppo_math.h"

// Batched PPO kernels. All are asynchronous on `stream`, deterministic (no
// atomics; reductions run in a fixed order), and throw cuda_error on launch failure.
namespace cudarl::rl {

// One action per row of logits [n x k]: sampled with uniforms from (seed, stream_id, row), or the argmax.
void sample_actions(const float* logits, int n, int k, uint32_t seed, uint32_t stream_id, bool greedy, int* actions,
                    float* logp, cudaStream_t stream);

// GAE over a [steps x num_envs] rollout, one thread per environment.
void compute_gae(int steps, int num_envs, const float* rewards, const float* values, const float* dones,
                 const float* truncated, const float* final_values, const float* last_values, float gamma,
                 float lambda, float* advantages, float* returns, cudaStream_t stream);

// Minibatch gather: for i in [0, count), row i of each output is row
// permute(perm, first + i) of the corresponding input. obs rows have obs_dim floats.
void gather_minibatch(const Permutation& perm, int first, int count, int obs_dim, const float* obs, const int* actions,
                      const float* logp, const float* advantages, const float* returns, float* mb_obs, int* mb_actions,
                      float* mb_logp, float* mb_advantages, float* mb_returns, cudaStream_t stream);

// out[0] = mean, out[1] = population standard deviation of x[0..n).
void mean_std(const float* x, int n, float* out, cudaStream_t stream);

// PPO loss gradient for a minibatch. Advantages are normalized with
// adv_stats = (mean, std) first. Writes dlogits [n x k], dvalues [n], and adds
// this minibatch's summed (pg_loss, v_loss, entropy, approx_kl, clipped) to stats[0..5).
constexpr int kPpoStats = 5;
void ppo_loss_grad(const float* logits, const float* values, int n, int k, const int* actions, const float* old_logp,
                   const float* advantages, const float* returns, const float* adv_stats, PpoCoefficients coeffs,
                   float* dlogits, float* dvalues, float* row_stats, float* stats, cudaStream_t stream);

// Episodes finished during a rollout: out[0] += sum of lengths, out[1] += count.
void episode_stats(const float* finished_length, int n, float* out, cudaStream_t stream);

} // namespace cudarl::rl

#endif // CUDARL_RL_PPO_KERNELS_H
