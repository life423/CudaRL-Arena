#ifndef CUDARL_RL_GAUSSIAN_MATH_H
#define CUDARL_RL_GAUSSIAN_MATH_H

#include <cmath>

#include "core/hd.h"
#include "rl/ppo_math.h"

// The per-element math of a diagonal Gaussian policy, shared by the CUDA kernels
// and the tests. The action is a vector; each dimension has its own standard
// deviation, exp(log_std), which does not depend on the state.
namespace cudarl::rl {

constexpr double kLog2Pi = 1.83787706640934548;   // log(2 pi)

// log N(action; mean, diag(exp(log_std))^2), summed over the dimensions.
template <typename T>
CUDARL_HD inline T gaussian_logp(const T* mean, const T* log_std, const T* action, int dim) {
    T logp = T(0);
    for (int j = 0; j < dim; ++j) {
        const T z = (action[j] - mean[j]) * exp(-log_std[j]);
        logp += T(-0.5) * z * z - log_std[j] - T(0.5) * T(kLog2Pi);
    }
    return logp;
}

// Entropy of the diagonal Gaussian: sum_j (log_std_j + 0.5 log(2 pi e)).
template <typename T>
CUDARL_HD inline T gaussian_entropy(const T* log_std, int dim) {
    T h = T(0);
    for (int j = 0; j < dim; ++j) h += log_std[j] + T(0.5) * (T(kLog2Pi) + T(1));
    return h;
}

// PPO loss for one sample under the Gaussian policy, with the same clipped
// surrogate, value loss and entropy bonus as ppo_row. The gradients of
// log pi(a) are d/d(mean_j) = (a_j - mean_j) / sigma_j^2 and
// d/d(log_std_j) = z_j^2 - 1 with z_j = (a_j - mean_j) / sigma_j. Writes
// d(loss)/d(mean) and d(loss)/d(log_std) for this sample, scaled by `scale`.
template <typename T>
CUDARL_HD inline PpoRowTerms<T> ppo_gaussian_row(const T* mean, const T* log_std, int dim, const T* action,
                                                 T old_logp, T advantage, T value, T target_return,
                                                 PpoCoefficients c, T scale, T* dmean, T* dlog_std, T* dvalue) {
    const T logp      = gaussian_logp(mean, log_std, action, dim);
    const T entropy   = gaussian_entropy(log_std, dim);
    const T log_ratio = logp - old_logp;
    const T ratio     = exp(log_ratio);
    const T lo = T(1) - T(c.clip), hi = T(1) + T(c.clip);
    const T clipped_ratio = ratio < lo ? lo : (ratio > hi ? hi : ratio);
    const T surr1 = ratio * advantage, surr2 = clipped_ratio * advantage;
    const bool unclipped_is_min = surr1 <= surr2;
    const T g_logp = unclipped_is_min ? -advantage * ratio : T(0);   // d(pg_loss)/d(log pi(a))

    for (int j = 0; j < dim; ++j) {
        const T inv_var = exp(-T(2) * log_std[j]);
        const T diff    = action[j] - mean[j];
        dmean[j]    = scale * g_logp * diff * inv_var;
        // The entropy term is -ent_coef * H, and dH/d(log_std_j) = 1.
        dlog_std[j] = scale * (g_logp * (diff * diff * inv_var - T(1)) - T(c.ent_coef));
    }
    const T v_err = value - target_return;
    *dvalue = scale * T(c.vf_coef) * v_err;

    PpoRowTerms<T> terms;
    terms.pg_loss   = -(unclipped_is_min ? surr1 : surr2);
    terms.v_loss    = T(0.5) * v_err * v_err;
    terms.entropy   = entropy;
    terms.approx_kl = (ratio - T(1)) - log_ratio;
    const T dev = ratio - T(1);
    terms.clipped = (dev > T(c.clip) || dev < -T(c.clip)) ? T(1) : T(0);
    return terms;
}

} // namespace cudarl::rl

#endif // CUDARL_RL_GAUSSIAN_MATH_H
