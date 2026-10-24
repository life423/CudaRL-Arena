#ifndef CUDARL_RL_PPO_MATH_H
#define CUDARL_RL_PPO_MATH_H

#include <cmath>
#include <cstdint>

#include "core/hd.h"
#include "core/rng.h"

// The per-element math of PPO, shared by the CUDA kernels and the tests.
namespace cudarl::rl {

constexpr int kMaxActions = 8;

// --- Categorical policy -------------------------------------------------------

// logp[0..k) = log softmax(logits), subtracting the max first so large logits don't overflow.
template <typename T>
CUDARL_HD inline void log_softmax(const T* logits, int k, T* logp) {
    T m = logits[0];
    for (int j = 1; j < k; ++j) m = logits[j] > m ? logits[j] : m;
    T sum = T(0);
    for (int j = 0; j < k; ++j) sum += exp(logits[j] - m);
    const T log_sum = log(sum) + m;
    for (int j = 0; j < k; ++j) logp[j] = logits[j] - log_sum;
}

// Inverse-CDF sample with a uniform u in (0, 1); greedy picks the argmax.
CUDARL_HD inline int sample_action(const float* logp, int k, float u, bool greedy) {
    if (greedy) {
        int best = 0;
        for (int j = 1; j < k; ++j) best = logp[j] > logp[best] ? j : best;
        return best;
    }
    float cdf = 0.f;
    for (int j = 0; j < k - 1; ++j) {
        cdf += expf(logp[j]);
        if (u < cdf) return j;
    }
    return k - 1;
}

// --- Random permutation without sorting -----------------------------------------

// A 4-round balanced Feistel network is a keyed bijection on [0, 4^b); "cycle
// walking" (re-applying it until the value lands below n) restricts it to a
// bijection on [0, n). Each index is permuted independently, so a minibatch
// gather needs no sort and no shared state.
struct Permutation {
    uint32_t n;
    int      half_bits;
    uint32_t mask;
    uint32_t keys[4];
};

CUDARL_HD inline Permutation make_permutation(uint32_t n, uint32_t seed) {
    Permutation p{};
    p.n = n;
    int bits = 2;
    while (bits < 30 && (1u << bits) < n) bits += 2;
    p.half_bits = bits / 2;
    p.mask      = (1u << p.half_bits) - 1u;
    for (uint32_t r = 0; r < 4; ++r) p.keys[r] = rng::hash_u32(seed ^ rng::hash_u32(r + 0x51ed270bu));
    return p;
}

CUDARL_HD inline uint32_t feistel(const Permutation& p, uint32_t x) {
    uint32_t left = x >> p.half_bits, right = x & p.mask;
    for (int r = 0; r < 4; ++r) {
        const uint32_t f = rng::hash_u32(right ^ p.keys[r]) & p.mask;
        const uint32_t next_left = right;
        right = left ^ f;
        left  = next_left;
    }
    return (left << p.half_bits) | right;
}

CUDARL_HD inline uint32_t permute(const Permutation& p, uint32_t i) {
    uint32_t x = i;
    do {
        x = feistel(p, x);
    } while (x >= p.n);
    return x;
}

// --- Generalized advantage estimation ----------------------------------------

// GAE for one environment over a rollout stored [T][N] (index t * N + env).
// An episode boundary at step t (done) stops the bootstrap chain. If it was a
// truncation (the time limit, not a failure), the next state is still worth
// something, so the target bootstraps from V(final observation) instead of 0.
template <typename T>
CUDARL_HD inline void gae_one_env(int steps, int num_envs, int env, const T* rewards, const T* values, const T* dones,
                                  const T* truncated, const T* final_values, const T* last_values, T gamma, T lambda,
                                  T* advantages, T* returns) {
    T gae = T(0);
    for (int t = steps - 1; t >= 0; --t) {
        const int i = t * num_envs + env;
        const bool done = dones[i] != T(0);
        T next_value;
        if (done) next_value = truncated[i] != T(0) ? final_values[i] : T(0);
        else      next_value = t == steps - 1 ? last_values[env] : values[i + num_envs];
        const T delta = rewards[i] + gamma * next_value - values[i];
        gae = delta + gamma * lambda * (done ? T(0) : T(1)) * gae;
        advantages[i] = gae;
        returns[i]    = gae + values[i];
    }
}

// --- PPO loss and its gradient ---------------------------------------------------

struct PpoCoefficients {
    float clip;       // epsilon in clip(ratio, 1 - eps, 1 + eps)
    float ent_coef;   // entropy bonus weight
    float vf_coef;    // value loss weight
};

template <typename T>
struct PpoRowTerms {
    T pg_loss;    // -min(r A, clip(r) A)
    T v_loss;     // 0.5 (V - R)^2
    T entropy;
    T approx_kl;  // (r - 1) - log r, a low-variance KL(old || new) estimate
    T clipped;    // 1 if |r - 1| > eps
};

// Loss for one sample: pg_loss - ent_coef * entropy + vf_coef * v_loss, times
// `scale` (1 / minibatch size). Writes d(loss)/d(logits) and d(loss)/d(value).
// The clipped surrogate's gradient through the ratio is r * A * d(log pi(a))
// when the unclipped term is the minimum, and 0 when clipping is active.
template <typename T>
CUDARL_HD inline PpoRowTerms<T> ppo_row(const T* logits, int k, int action, T old_logp, T advantage, T value,
                                        T target_return, PpoCoefficients c, T scale, T* dlogits, T* dvalue) {
    T logp[kMaxActions];
    log_softmax(logits, k, logp);
    T entropy = T(0);
    for (int j = 0; j < k; ++j) entropy -= exp(logp[j]) * logp[j];

    const T log_ratio = logp[action] - old_logp;
    const T ratio     = exp(log_ratio);
    const T lo = T(1) - T(c.clip), hi = T(1) + T(c.clip);
    const T clipped_ratio = ratio < lo ? lo : (ratio > hi ? hi : ratio);
    const T surr1 = ratio * advantage, surr2 = clipped_ratio * advantage;
    const bool unclipped_is_min = surr1 <= surr2;
    const T g_logp = unclipped_is_min ? -advantage * ratio : T(0);   // d(pg_loss)/d(log pi(a))

    for (int j = 0; j < k; ++j) {
        const T p = exp(logp[j]);
        const T d_pg  = g_logp * ((j == action ? T(1) : T(0)) - p);
        const T d_ent = T(c.ent_coef) * p * (logp[j] + entropy);   // d(-ent_coef * H)/d(logit_j)
        dlogits[j] = scale * (d_pg + d_ent);
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

#endif // CUDARL_RL_PPO_MATH_H
