// PPO building blocks against independent references, then the full trainer:
// sampling statistics, the Feistel shuffle, GAE with truncation, the loss
// gradient vs finite differences, orthogonal init, and end-to-end learning.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "gpu_util.h"
#include "nn/mlp_spec.h"
#include "rl/ppo.h"
#include "rl/ppo_kernels.h"
#include "rl/ppo_math.h"

using namespace cudarl;
using namespace cudarl::rl;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

// --- Shuffle -----------------------------------------------------------------

TEST_CASE("ppo: Feistel shuffle is a permutation for any size") {
    for (uint32_t n : {1u, 2u, 3u, 7u, 1000u, 4096u, 100003u}) {
        const Permutation p = make_permutation(n, 1234);
        std::vector<int> hits(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t j = permute(p, i);
            CHECK(j < n);
            if (j < n) ++hits[j];
        }
        CHECK(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; }));
    }
}

TEST_CASE("ppo: Feistel shuffle depends on the seed and actually shuffles") {
    const uint32_t n = 4096;
    const Permutation a = make_permutation(n, 1), b = make_permutation(n, 2);
    int same = 0, fixed = 0;
    for (uint32_t i = 0; i < n; ++i) {
        same  += permute(a, i) == permute(b, i);
        fixed += permute(a, i) == i;
    }
    CHECK(same < 20);    // independent permutations agree on ~1 position
    CHECK(fixed < 20);
}

// --- Sampling ----------------------------------------------------------------

GPU_TEST_CASE("ppo: sampled action frequencies match the policy") {
    const int n = 200000, k = 3;
    std::vector<float> logits(static_cast<size_t>(n) * k);
    for (int i = 0; i < n; ++i) {
        logits[i * 3 + 0] = 0.3f;
        logits[i * 3 + 1] = -0.8f;
        logits[i * 3 + 2] = 1.1f;
    }
    auto d_logits = to_device(logits);
    device_ptr<int> actions(n);
    device_ptr<float> logp(n);
    sample_actions(d_logits.get(), n, k, 7, 0, false, actions.get(), logp.get(), nullptr);
    std::vector<int> a(n);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    actions.copy_to_host(a.data(), n);
    const auto lp = to_host(logp, n);

    double z = std::exp(0.3) + std::exp(-0.8) + std::exp(1.1);
    const double p[3] = {std::exp(0.3) / z, std::exp(-0.8) / z, std::exp(1.1) / z};
    int counts[3] = {};
    for (int i = 0; i < n; ++i) ++counts[a[i]];
    for (int j = 0; j < k; ++j) {
        const double sigma = std::sqrt(p[j] * (1 - p[j]) / n);
        CHECK_MSG(std::fabs(counts[j] / double(n) - p[j]) < 5 * sigma, "action " + std::to_string(j));
    }
    for (int i = 0; i < 100; ++i) CHECK(std::fabs(lp[i] - std::log(p[a[i]])) < 1e-5);

    sample_actions(d_logits.get(), n, k, 7, 0, true, actions.get(), logp.get(), nullptr);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    actions.copy_to_host(a.data(), n);
    CHECK(std::all_of(a.begin(), a.end(), [](int x) { return x == 2; }));   // greedy = argmax
}

// --- GAE -----------------------------------------------------------------------

namespace {

// Textbook GAE, written out independently of gae_one_env.
std::vector<double> reference_gae(int steps, int n, const std::vector<float>& r, const std::vector<float>& v,
                                  const std::vector<float>& done, const std::vector<float>& trunc,
                                  const std::vector<float>& vfinal, const std::vector<float>& vlast, double gamma,
                                  double lambda) {
    std::vector<double> adv(r.size());
    for (int e = 0; e < n; ++e) {
        double running = 0.0;
        for (int t = steps - 1; t >= 0; --t) {
            const int i = t * n + e;
            double bootstrap;
            if (done[i] > 0.5f) bootstrap = trunc[i] > 0.5f ? vfinal[i] : 0.0;
            else bootstrap = t + 1 < steps ? v[i + n] : vlast[e];
            const double delta = r[i] + gamma * bootstrap - v[i];
            running = delta + (done[i] > 0.5f ? 0.0 : gamma * lambda * running);
            adv[i] = running;
        }
    }
    return adv;
}

} // namespace

TEST_CASE("ppo: GAE on a hand-computed example") {
    // One env, 3 steps, terminating at t=1: the chain restarts there.
    const float r[3] = {1, 1, 1}, v[3] = {0.5f, 0.25f, 2.f}, done[3] = {0, 1, 0}, trunc[3] = {0, 0, 0};
    const float vfinal[3] = {0, 0, 0}, vlast[1] = {3.f};
    float adv[3], ret[3];
    gae_one_env<float>(3, 1, 0, r, v, done, trunc, vfinal, vlast, 0.9f, 0.8f, adv, ret);
    const float d2 = 1 + 0.9f * 3.f - 2.f;          // t=2 bootstraps from the value after the rollout
    const float d1 = 1 + 0.f - 0.25f;               // t=1 terminated: no bootstrap, chain cut
    const float d0 = 1 + 0.9f * 0.25f - 0.5f;
    CHECK(std::fabs(adv[2] - d2) < 1e-6);
    CHECK(std::fabs(adv[1] - d1) < 1e-6);
    CHECK(std::fabs(adv[0] - (d0 + 0.9f * 0.8f * d1)) < 1e-6);
    CHECK(std::fabs(ret[0] - (adv[0] + 0.5f)) < 1e-6);
}

TEST_CASE("ppo: GAE bootstraps truncated episodes from the final value") {
    const float r[1] = {1}, v[1] = {4.f}, done[1] = {1}, trunc[1] = {1}, vfinal[1] = {10.f}, vlast[1] = {-99.f};
    float adv[1], ret[1];
    gae_one_env<float>(1, 1, 0, r, v, done, trunc, vfinal, vlast, 0.99f, 0.95f, adv, ret);
    CHECK(std::fabs(adv[0] - (1 + 0.99f * 10.f - 4.f)) < 1e-5);   // not 1 - 4 (failure) and not using vlast
}

GPU_TEST_CASE("ppo: GPU GAE matches the reference with random episode ends") {
    const int steps = 64, n = 777;
    const size_t b = size_t(steps) * n;
    auto r = random_floats(b, 1), v = random_floats(b, 2), vfinal = random_floats(b, 3), vlast = random_floats(n, 4);
    std::vector<float> done(b), trunc(b);
    for (size_t i = 0; i < b; ++i) {
        done[i]  = rng::uniform(5, 0, static_cast<uint32_t>(i)) < 0.08f ? 1.f : 0.f;
        trunc[i] = done[i] > 0 && rng::uniform(6, 0, static_cast<uint32_t>(i)) < 0.3f ? 1.f : 0.f;
    }
    auto dr = to_device(r), dv = to_device(v), dd = to_device(done), dt = to_device(trunc), dvf = to_device(vfinal),
         dvl = to_device(vlast);
    device_ptr<float> adv(b), ret(b);
    compute_gae(steps, n, dr.get(), dv.get(), dd.get(), dt.get(), dvf.get(), dvl.get(), 0.99f, 0.95f, adv.get(),
                ret.get(), nullptr);
    const auto want = reference_gae(steps, n, r, v, done, trunc, vfinal, vlast, 0.99, 0.95);
    CHECK_CLOSE(to_host(adv, b), want, 1e-5);
}

// --- Loss gradient -------------------------------------------------------------

namespace {

// The PPO per-sample loss, written independently in double for finite differences.
double reference_loss(const double* logits, int k, int a, double old_logp, double adv, double value, double ret,
                      PpoCoefficients c) {
    double m = logits[0];
    for (int j = 1; j < k; ++j) m = std::max(m, logits[j]);
    double z = 0.0;
    for (int j = 0; j < k; ++j) z += std::exp(logits[j] - m);
    double entropy = 0.0;
    for (int j = 0; j < k; ++j) {
        const double p = std::exp(logits[j] - m) / z;
        entropy -= p * std::log(p);
    }
    const double ratio = std::exp(logits[a] - m - std::log(z) - old_logp);
    const double clipped = std::min(std::max(ratio, 1.0 - c.clip), 1.0 + c.clip);
    const double pg = -std::min(ratio * adv, clipped * adv);
    return pg - c.ent_coef * entropy + c.vf_coef * 0.5 * (value - ret) * (value - ret);
}

void check_gradient(const std::vector<double>& logits, int a, double old_logp, double adv, double value, double ret,
                    const char* label) {
    const PpoCoefficients c{0.2f, 0.01f, 0.5f};
    const int k = static_cast<int>(logits.size());
    double dl[kMaxActions], dv;
    ppo_row<double>(logits.data(), k, a, old_logp, adv, value, ret, c, 1.0, dl, &dv);

    const double h = 1e-6;
    auto lp = logits;
    for (int j = 0; j < k; ++j) {
        lp[j] = logits[j] + h;
        const double up = reference_loss(lp.data(), k, a, old_logp, adv, value, ret, c);
        lp[j] = logits[j] - h;
        const double down = reference_loss(lp.data(), k, a, old_logp, adv, value, ret, c);
        lp[j] = logits[j];
        const double numeric = (up - down) / (2 * h);
        CHECK_MSG(std::fabs(dl[j] - numeric) < 1e-7, std::string(label) + ": dlogit " + std::to_string(j));
    }
    const double num_v = (reference_loss(logits.data(), k, a, old_logp, adv, value + h, ret, c) -
                          reference_loss(logits.data(), k, a, old_logp, adv, value - h, ret, c)) / (2 * h);
    CHECK_MSG(std::fabs(dv - num_v) < 1e-7, std::string(label) + ": dvalue");
}

} // namespace

TEST_CASE("ppo: loss gradient matches finite differences in every regime") {
    const std::vector<double> logits{0.2, -0.4, 0.9};
    // log pi(a=0) under these logits is about -1.27.
    check_gradient(logits, 0, -1.27, 1.3, 0.4, 1.1, "ratio ~ 1, positive advantage");
    check_gradient(logits, 0, -1.27, -0.7, 0.4, -0.2, "ratio ~ 1, negative advantage");
    check_gradient(logits, 0, -1.9, 1.3, 0.4, 1.1, "ratio > 1 + eps, A > 0 (clipped: entropy only)");
    check_gradient(logits, 0, -1.9, -0.7, 0.4, 1.1, "ratio > 1 + eps, A < 0 (unclipped)");
    check_gradient(logits, 2, -0.2, -0.9, -1.0, 0.3, "ratio < 1 - eps, A < 0 (clipped)");
    check_gradient(logits, 2, -0.2, 0.9, -1.0, 0.3, "ratio < 1 - eps, A > 0 (unclipped)");
    check_gradient({3.0, -3.0}, 1, -6.0, 0.5, 2.0, 2.0, "two actions, near-deterministic policy");
}

GPU_TEST_CASE("ppo: GPU loss gradient kernel matches the host math") {
    const int n = 3000, k = 2;
    auto logits = random_floats(size_t(n) * k, 11), values = random_floats(n, 12), adv = random_floats(n, 13, 2.f),
         ret = random_floats(n, 14), old = random_floats(n, 15, 0.3f);
    std::vector<int> actions(n);
    for (int i = 0; i < n; ++i) {
        actions[i] = static_cast<int>(rng::bits(16, 0, i) & 1u);
        old[i] -= 0.7f;   // around log(1/2)
    }
    auto dl = to_device(logits), dv = to_device(values), da = to_device(adv), dr = to_device(ret), dold = to_device(old);
    device_ptr<int> dact(n);
    dact.copy_from_host(actions.data(), n);
    device_ptr<float> stats2(2), dlogits(size_t(n) * k), dvalues(n), rows(size_t(n) * kPpoStats), sums(kPpoStats);
    CUDA_CHECK_THROW(cudaMemset(sums.get(), 0, sums.size_bytes()));
    mean_std(da.get(), n, stats2.get(), nullptr);
    const PpoCoefficients c{0.2f, 0.01f, 0.5f};
    ppo_loss_grad(dl.get(), dv.get(), n, k, dact.get(), dold.get(), da.get(), dr.get(), stats2.get(), c, dlogits.get(),
                  dvalues.get(), rows.get(), sums.get(), nullptr);

    // Host: same normalization, same per-row math.
    double mean = 0, var = 0;
    for (float a : adv) mean += a;
    mean /= n;
    for (float a : adv) var += (a - mean) * (a - mean);
    const double sd = std::sqrt(var / n);
    const auto ms = to_host(stats2, 2);
    CHECK(std::fabs(ms[0] - mean) < 1e-5 && std::fabs(ms[1] - sd) < 1e-4);
    std::vector<float> want_dl(size_t(n) * k), want_dv(n);
    double want_kl = 0;
    for (int i = 0; i < n; ++i) {
        const float a = static_cast<float>((adv[i] - mean) / (sd + 1e-8));
        const auto t = ppo_row<float>(&logits[i * k], k, actions[i], old[i], a, values[i], ret[i], c, 1.f / n,
                                      &want_dl[i * k], &want_dv[i]);
        want_kl += t.approx_kl;
    }
    CHECK_CLOSE(to_host(dlogits, want_dl.size()), want_dl, 1e-4);
    CHECK_CLOSE(to_host(dvalues, n), want_dv, 1e-5);
    CHECK(std::fabs(to_host(sums, kPpoStats)[3] - want_kl) < 1e-3 * std::max(1.0, std::fabs(want_kl)));
}

// --- Orthogonal init -------------------------------------------------------------

TEST_CASE("ppo: orthogonal init gives scaled orthonormal weights") {
    const nn::MlpSpec spec({4, 64, 64, 2}, {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity});
    const std::vector<float> gains{1.41421356f, 1.41421356f, 0.01f};
    const auto p = nn::orthogonal_params(spec, 3, gains);
    for (int l = 0; l < spec.num_layers(); ++l) {
        const int rows = spec.out(l), cols = spec.in(l);
        const float* W = &p[spec.weight_offset(l)];
        const bool by_columns = rows >= cols;
        const int count = by_columns ? cols : rows, dim = by_columns ? rows : cols;
        double worst = 0;
        for (int u = 0; u < count; ++u)
            for (int v = 0; v < count; ++v) {
                double dot = 0;
                for (int d = 0; d < dim; ++d) {
                    const double a = by_columns ? W[d * cols + u] : W[u * cols + d];
                    const double b = by_columns ? W[d * cols + v] : W[v * cols + d];
                    dot += a * b;
                }
                const double want = u == v ? double(gains[l]) * gains[l] : 0.0;
                worst = std::max(worst, std::fabs(dot - want) / (double(gains[l]) * gains[l]));
            }
        CHECK_MSG(worst < 1e-5, "layer " + std::to_string(l) + " orthonormality error " + std::to_string(worst));
        for (int b = spec.bias_offset(l); b < spec.bias_offset(l) + rows; ++b) CHECK(p[b] == 0.f);
    }
}

// --- End to end ---------------------------------------------------------------------

GPU_TEST_CASE("ppo: solves CartPole (mean episode length >= 475) on 3 seeds") {
    for (uint32_t seed : {1u, 2u, 3u}) {
        PpoConfig cfg;
        cfg.seed = seed;
        PpoTrainer trainer(cfg);
        int streak = 0, updates = 0;
        while (streak < 3 && updates < 60) {
            streak = trainer.update().mean_episode_length >= 475.f ? streak + 1 : 0;
            ++updates;
        }
        CHECK_MSG(streak == 3, "seed " + std::to_string(seed) + " not solved in 60 updates");
        std::printf("    seed %u solved after %d updates (%d env steps)\n", seed, updates,
                    updates * cfg.num_envs * cfg.rollout_steps);
    }
}

GPU_TEST_CASE("ppo: training is bit-identical for the same seed") {
    PpoConfig cfg;
    cfg.num_envs = 256;
    cfg.rollout_steps = 16;
    std::vector<float> first_params;
    std::vector<float> first_curve;
    for (int run = 0; run < 2; ++run) {
        PpoTrainer trainer(cfg);
        std::vector<float> curve;
        for (int u = 0; u < 5; ++u) {
            const auto s = trainer.update();
            curve.push_back(s.mean_episode_length);
            curve.push_back(s.approx_kl);
        }
        if (run == 0) {
            first_params = trainer.actor_params();
            first_curve = curve;
        } else {
            CHECK(trainer.actor_params() == first_params);
            CHECK(curve == first_curve);
        }
    }
}
