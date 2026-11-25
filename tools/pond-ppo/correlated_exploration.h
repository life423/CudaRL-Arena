#pragma once

#include "nn/mlp_gpu.h"
#include "rl/ar_gaussian_math.h"
#include "rl/gaussian_kernels.h"
#include "rl/pond_ppo_contract.h"

namespace cudarl::rl::pond_correlated {

namespace contract = pond_ppo;
inline constexpr float rho = contract::correlated_rho;
inline constexpr int width = contract::signed_inputs;
inline constexpr int actions = contract::actions;
inline constexpr int worlds = contract::worlds;
inline constexpr int capacity = contract::minibatch;
inline constexpr int rollout_rows = worlds * contract::controls_per_world;

namespace detail {
// Columns within a row are contiguous. The network sees a batch of 2*n rows:
// all current observations followed by their row-matched previous observations.
static __global__ void pack_pairs(const float* observations, const float* latents, const float* done,
                                  int first, int n, Permutation order, float* paired,
                                  float* previous_action, float* has_history) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * width) return;
    const int i = index / width, column = index % width;
    const int row = order.n ? static_cast<int>(permute(order, first + i)) : first + i;
    const int previous = row - worlds;
    const bool valid = previous >= 0 && done[previous] == 0.f;
    paired[index] = observations[row * width + column];
    paired[n * width + index] = valid ? observations[previous * width + column] : 0.f;
    if (column < actions)
        previous_action[i * actions + column] = valid ? latents[previous * actions + column] : 0.f;
    if (column == 0) has_history[i] = valid ? 1.f : 0.f;
}

static __global__ void check_probabilities(const float* means, const float* log_std,
                                          const float* previous_action, const float* has_history,
                                          const float* latent, const float* old_logp, int n,
                                          int first, float* kl, int* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float actual = ar_logp(means + i * actions, means + (n + i) * actions,
                                 previous_action + i * actions, log_std, actions, rho,
                                 has_history[i] != 0.f, latent + i * actions);
    const float delta = actual - old_logp[i];
    if (!isfinite(delta)) { atomicMin(fault, first + i); return; }
    if (kl) {
        kl[i] = expf(delta) - 1.f - delta;
        if (!isfinite(kl[i])) atomicMin(fault, first + i);
    } else if (fabsf(delta) > 1e-4f) {
        atomicMin(fault, first + i);
    }
}
} // namespace detail

class Policy {
public:
    // An omitted order addresses consecutive rows. Training passes the exact
    // same permutation that gathered the current action, advantage and return.
    void forward(nn::GpuMlp& actor, const float* observations, const float* latents, const float* done,
                 int first, int n, cudaStream_t stream, const Permutation* order = nullptr) {
        if (n < 1 || n > capacity || first < 0 || first > rollout_rows - n ||
            (order && order->n != static_cast<uint32_t>(rollout_rows)))
            throw std::invalid_argument("Invalid correlated-exploration row range");
        detail::pack_pairs<<<(n * width + 255) / 256, 256, 0, stream>>>(
            observations, latents, done, first, n, order ? *order : Permutation{},
            paired_.get(), previous_action_.get(), history_.get());
        CUDA_CHECK_THROW(cudaGetLastError());
        actor.forward(paired_.get(), 2 * n, stream);
    }

    void sample(const nn::GpuMlp& actor, const float* log_std, int n, uint32_t seed,
                uint32_t serial, float* latent, float* logp, cudaStream_t stream) const {
        sample_ar_gaussian(actor.output(), log_std, previous_action_.get(), history_.get(),
                           n, actions, rho, seed, serial, latent, logp, stream);
    }

    // Pass pointers to the first contiguous action/logp in this audit chunk.
    // A null KL destination checks that the stored collection logp is exact
    // to the existing tolerance. A destination writes the post-update KL rows.
    void audit(const nn::GpuMlp& actor, const float* log_std, const float* latent,
               const float* old_logp, int n, int first, float* kl, int* fault,
               cudaStream_t stream) const {
        detail::check_probabilities<<<(n + 255) / 256, 256, 0, stream>>>(
            actor.output(), log_std, previous_action_.get(), history_.get(),
            latent, old_logp, n, first, kl, fault);
        CUDA_CHECK_THROW(cudaGetLastError());
    }

    void loss(const nn::GpuMlp& actor, const float* log_std, const float* values, int n,
              const float* latent, const float* old_logp, const float* advantages, const float* returns,
              const float* adv_stats, PpoCoefficients coefficients, float* dvalue, float* row_stats,
              float* stats, cudaStream_t stream) {
        ar_gaussian_ppo_loss_grad(actor.output(), log_std, previous_action_.get(), history_.get(),
                                  values, n, actions, rho, latent, old_logp, advantages, returns,
                                  adv_stats, coefficients, dmean_.get(), dvalue, row_stats, stats, stream);
    }

    const float* gradients() const { return dmean_.get(); }

private:
    device_ptr<float> paired_{2 * capacity * width};
    device_ptr<float> previous_action_{capacity * actions};
    device_ptr<float> history_{capacity};
    device_ptr<float> dmean_{2 * capacity * actions};
};
} // namespace cudarl::rl::pond_correlated
