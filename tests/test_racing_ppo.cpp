// The Gaussian policy: log-probabilities and sampling statistics, the PPO loss
// gradient against finite differences, the GPU kernels against the host math,
// and the racing trainer end to end (determinism, and whether the cars learn).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "gpu_util.h"
#include "rl/gaussian_kernels.h"
#include "rl/gaussian_math.h"
#include "rl/ppo_kernels.h"
#include "rl/ppo_math.h"
#include "rl/racing_ppo.h"

using namespace cudarl;
using namespace cudarl::rl;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

namespace {

constexpr double kPiD     = 3.14159265358979323846;
constexpr double kEulerD  = 2.71828182845904523536;

// Independent double-precision references, written out from the definitions.
double ref_logp(const std::vector<double>& mean, const std::vector<double>& log_std, const std::vector<double>& a) {
    const double log2pi = std::log(2.0 * kPiD);
    double logp = 0.0;
    for (size_t j = 0; j < mean.size(); ++j) {
        const double z = (a[j] - mean[j]) / std::exp(log_std[j]);
        logp += -0.5 * z * z - log_std[j] - 0.5 * log2pi;
    }
    return logp;
}

double ref_entropy(const std::vector<double>& log_std) {
    const double log2pi = std::log(2.0 * kPiD);
    double h = 0.0;
    for (double ls : log_std) h += ls + 0.5 * (log2pi + 1.0);
    return h;
}

double ref_loss(const std::vector<double>& mean, const std::vector<double>& log_std, const std::vector<double>& a,
                double old_logp, double adv, double value, double ret, PpoCoefficients c) {
    const double ratio   = std::exp(ref_logp(mean, log_std, a) - old_logp);
    const double clipped = std::min(std::max(ratio, 1.0 - c.clip), 1.0 + c.clip);
    const double pg      = -std::min(ratio * adv, clipped * adv);
    return pg - c.ent_coef * ref_entropy(log_std) + c.vf_coef * 0.5 * (value - ret) * (value - ret);
}

// Compares the analytic gradient with central differences in every input.
void check_gaussian_gradient(std::vector<double> mean, std::vector<double> log_std, const std::vector<double>& a,
                             double ratio_log, double adv, double value, double ret, const char* label) {
    const PpoCoefficients c{0.2f, 0.05f, 0.5f};
    const int dim = static_cast<int>(mean.size());
    const double logp = ref_logp(mean, log_std, a);
    const double old_logp = logp - ratio_log;   // so that ratio = exp(ratio_log)

    std::vector<double> dm(dim), dls(dim);
    double dv = 0.0;
    ppo_gaussian_row<double>(mean.data(), log_std.data(), dim, a.data(), old_logp, adv, value, ret, c, 1.0,
                             dm.data(), dls.data(), &dv);

    const double h = 1e-6;
    for (int j = 0; j < dim; ++j) {
        auto m = mean;
        m[j] = mean[j] + h;
        const double up = ref_loss(m, log_std, a, old_logp, adv, value, ret, c);
        m[j] = mean[j] - h;
        const double down = ref_loss(m, log_std, a, old_logp, adv, value, ret, c);
        CHECK_MSG(std::fabs(dm[j] - (up - down) / (2 * h)) < 1e-7, std::string(label) + ": dmean " + std::to_string(j));

        auto s = log_std;
        s[j] = log_std[j] + h;
        const double up_s = ref_loss(mean, s, a, old_logp, adv, value, ret, c);
        s[j] = log_std[j] - h;
        const double down_s = ref_loss(mean, s, a, old_logp, adv, value, ret, c);
        CHECK_MSG(std::fabs(dls[j] - (up_s - down_s) / (2 * h)) < 1e-7,
                  std::string(label) + ": dlog_std " + std::to_string(j));
    }
    const double num_v = (ref_loss(mean, log_std, a, old_logp, adv, value + h, ret, c) -
                          ref_loss(mean, log_std, a, old_logp, adv, value - h, ret, c)) / (2 * h);
    CHECK_MSG(std::fabs(dv - num_v) < 1e-7, std::string(label) + ": dvalue");
}

} // namespace

// --- Policy math -----------------------------------------------------------------

TEST_CASE("gaussian: log-probability and entropy match the closed forms") {
    const std::vector<double> mean{0.3, -0.7}, log_std{-0.4, 0.2}, a{0.9, -1.1};
    // One dimension at a time, against the scalar normal density.
    const double sigma0 = std::exp(-0.4), sigma1 = std::exp(0.2);
    const double lp0 = -0.5 * std::pow((0.9 - 0.3) / sigma0, 2) - std::log(sigma0) - 0.5 * std::log(2 * kPiD);
    const double lp1 = -0.5 * std::pow((-1.1 + 0.7) / sigma1, 2) - std::log(sigma1) - 0.5 * std::log(2 * kPiD);
    CHECK(std::fabs(gaussian_logp<double>(mean.data(), log_std.data(), a.data(), 2) - (lp0 + lp1)) < 1e-12);
    // Entropy of a standard normal is 0.5 log(2 pi e) per dimension.
    const double std_normal = 0.5 * std::log(2 * kPiD * kEulerD);
    const std::vector<double> zero{0.0, 0.0};
    CHECK(std::fabs(gaussian_entropy<double>(zero.data(), 2) - 2 * std_normal) < 1e-12);
    CHECK(std::fabs(gaussian_entropy<double>(log_std.data(), 2) - ref_entropy(log_std)) < 1e-12);
}

TEST_CASE("gaussian: loss gradient matches finite differences in every regime") {
    const std::vector<double> mean{0.2, -0.4}, log_std{-0.3, 0.1}, a{0.5, -0.9};
    check_gaussian_gradient(mean, log_std, a, 0.0, 1.3, 0.4, 1.1, "ratio 1, positive advantage");
    check_gaussian_gradient(mean, log_std, a, 0.0, -0.7, 0.4, -0.2, "ratio 1, negative advantage");
    check_gaussian_gradient(mean, log_std, a, 0.5, 1.3, 0.4, 1.1, "ratio above 1 + eps, A > 0 (clipped)");
    check_gaussian_gradient(mean, log_std, a, 0.5, -0.7, 0.4, 1.1, "ratio above 1 + eps, A < 0 (unclipped)");
    check_gaussian_gradient(mean, log_std, a, -0.5, -0.9, -1.0, 0.3, "ratio below 1 - eps, A < 0 (clipped)");
    check_gaussian_gradient(mean, log_std, a, -0.5, 0.9, -1.0, 0.3, "ratio below 1 - eps, A > 0 (unclipped)");
    check_gaussian_gradient({0.1, 0.0, -0.2}, {-1.0, 0.5, 0.0}, {0.3, 0.2, -0.1}, 0.05, 0.8, 0.2, 0.5,
                            "three dimensions");
}

// --- GPU kernels -----------------------------------------------------------------

GPU_TEST_CASE("gaussian: sampled actions have the policy's mean and spread") {
    const int n = 200000, dim = 2;
    const std::vector<float> mean_host{0.5f, -1.0f}, log_std_host{-0.5f, 0.3f};
    std::vector<float> mean(static_cast<size_t>(n) * dim);
    for (int i = 0; i < n; ++i) {
        mean[i * dim]     = mean_host[0];
        mean[i * dim + 1] = mean_host[1];
    }
    auto d_mean = to_device(mean);
    auto d_log_std = to_device(log_std_host);
    device_ptr<float> actions(mean.size()), logp(n);
    sample_gaussian(d_mean.get(), d_log_std.get(), n, dim, 7, 0, actions.get(), logp.get(), nullptr);
    const auto a = to_host(actions, mean.size());
    const auto lp = to_host(logp, n);

    for (int j = 0; j < dim; ++j) {
        double sum = 0, sq = 0;
        for (int i = 0; i < n; ++i) {
            sum += a[i * dim + j];
            sq += double(a[i * dim + j]) * a[i * dim + j];
        }
        const double m = sum / n, var = sq / n - m * m;
        const double sigma = std::exp(double(log_std_host[j]));
        CHECK_MSG(std::fabs(m - mean_host[j]) < 5 * sigma / std::sqrt(double(n)), "mean of dim " + std::to_string(j));
        CHECK_MSG(std::fabs(std::sqrt(var) / sigma - 1.0) < 0.01, "spread of dim " + std::to_string(j));
    }
    // The returned log-probabilities are those of the drawn actions.
    for (int i = 0; i < 100; ++i) {
        const std::vector<double> m{mean_host[0], mean_host[1]}, s{log_std_host[0], log_std_host[1]};
        const std::vector<double> x{a[i * dim], a[i * dim + 1]};
        CHECK(std::fabs(lp[i] - ref_logp(m, s, x)) < 1e-3);
    }
}

GPU_TEST_CASE("gaussian: GPU loss gradient matches the host math") {
    const int n = 3000, dim = 2;
    auto mean = random_floats(size_t(n) * dim, 31, 0.5f), values = random_floats(n, 32), adv = random_floats(n, 33, 2.f),
         ret = random_floats(n, 34);
    std::vector<float> log_std{-0.3f, 0.2f}, actions(size_t(n) * dim), old(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            const auto idx = static_cast<uint32_t>(i * dim + j);
            actions[i * dim + j] = mean[i * dim + j] + std::exp(log_std[j]) * rng::normal(35, 0, idx);
        }
        const std::vector<double> m{mean[i * dim], mean[i * dim + 1]}, s{log_std[0], log_std[1]};
        const std::vector<double> x{actions[i * dim], actions[i * dim + 1]};
        old[i] = static_cast<float>(ref_logp(m, s, x) - 0.1 * rng::uniform(36, 0, static_cast<uint32_t>(i)));
    }
    auto d_mean = to_device(mean), d_ls = to_device(log_std), d_val = to_device(values), d_act = to_device(actions),
         d_old = to_device(old), d_adv = to_device(adv), d_ret = to_device(ret);
    device_ptr<float> stats2(2), dmean(size_t(n) * dim), dls_rows(size_t(n) * dim), dvalues(n),
        rows(size_t(n) * kGaussianStats), sums(kGaussianStats);
    CUDA_CHECK_THROW(cudaMemset(sums.get(), 0, sums.size_bytes()));
    mean_std(d_adv.get(), n, stats2.get(), nullptr);
    const PpoCoefficients c{0.2f, 0.05f, 0.5f};
    gaussian_ppo_loss_grad(d_mean.get(), d_ls.get(), d_val.get(), n, dim, d_act.get(), d_old.get(), d_adv.get(),
                           d_ret.get(), stats2.get(), c, dmean.get(), dls_rows.get(), dvalues.get(), rows.get(),
                           sums.get(), nullptr);

    double mu = 0, var = 0;
    for (float a : adv) mu += a;
    mu /= n;
    for (float a : adv) var += (a - mu) * (a - mu);
    const double sd = std::sqrt(var / n);

    std::vector<float> want_dm(size_t(n) * dim), want_dls(size_t(n) * dim), want_dv(n);
    double want_kl = 0;
    for (int i = 0; i < n; ++i) {
        const float a = static_cast<float>((adv[i] - mu) / (sd + 1e-8));
        const auto t = ppo_gaussian_row<float>(&mean[i * dim], log_std.data(), dim, &actions[i * dim], old[i], a,
                                               values[i], ret[i], c, 1.f / n, &want_dm[i * dim], &want_dls[i * dim],
                                               &want_dv[i]);
        want_kl += t.approx_kl;
    }
    CHECK_CLOSE(to_host(dmean, want_dm.size()), want_dm, 1e-4);
    CHECK_CLOSE(to_host(dls_rows, want_dls.size()), want_dls, 1e-4);
    CHECK_CLOSE(to_host(dvalues, n), want_dv, 1e-5);
    const auto totals = to_host(sums, kGaussianStats);
    CHECK(std::fabs(totals[3] - want_kl) < 1e-3 * std::max(1.0, std::fabs(want_kl)));
    // The per-dimension log-std gradient is the sum over rows of the per-row gradients.
    std::vector<float> sum_ls(dim, 0.f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) sum_ls[j] += want_dls[i * dim + j];
    }
    device_ptr<float> shared(dim);
    CUDA_CHECK_THROW(cudaMemset(shared.get(), 0, shared.size_bytes()));
    sum_columns(dls_rows.get(), n, dim, shared.get(), nullptr);
    CHECK_CLOSE(to_host(shared, dim), sum_ls, 1e-4);
}

GPU_TEST_CASE("gaussian: gather_rows follows the permutation and episode_totals sums correctly") {
    const int rows = 1000, width = 3, first = 300, count = 200;
    const auto src_host = random_floats(size_t(rows) * width, 41);
    auto src = to_device(src_host);
    device_ptr<float> dst(size_t(count) * width);
    const Permutation perm = make_permutation(rows, 9);
    gather_rows(perm, first, count, width, src.get(), dst.get(), nullptr);
    const auto got = to_host(dst, size_t(count) * width);
    bool ok = true;
    for (int i = 0; i < count && ok; ++i) {
        for (int c = 0; c < width; ++c) {
            ok = ok && got[i * width + c] == src_host[size_t(permute(perm, first + i)) * width + c];
        }
    }
    CHECK(ok);

    const int n = 37;
    std::vector<float> len(n, 0.f), laps(n, 0.f), done(n, 0.f), trunc(n, 0.f);
    double want_len = 0, want_laps = 0, want_ended = 0, want_crash = 0;
    for (int i = 0; i < n; ++i) {
        const bool ended = rng::uniform(42, 0, i) < 0.5f;
        if (!ended) continue;
        const bool truncated = rng::uniform(43, 0, i) < 0.3f;
        len[i]   = 10.f + i;
        laps[i]  = 0.1f * i;
        done[i]  = 1.f;
        trunc[i] = truncated ? 1.f : 0.f;
        want_len += len[i];
        want_laps += laps[i];
        want_ended += 1;
        want_crash += truncated ? 0 : 1;
    }
    auto d_len = to_device(len), d_laps = to_device(laps), d_done = to_device(done), d_trunc = to_device(trunc);
    device_ptr<float> totals(4);
    CUDA_CHECK_THROW(cudaMemset(totals.get(), 0, totals.size_bytes()));
    episode_totals(d_len.get(), d_laps.get(), d_done.get(), d_trunc.get(), n, totals.get(), nullptr);
    const auto t = to_host(totals, 4);
    CHECK(std::fabs(t[0] - want_len) < 1e-3);
    CHECK(std::fabs(t[1] - want_laps) < 1e-3);
    CHECK(std::fabs(t[2] - want_ended) < 1e-6);
    CHECK(std::fabs(t[3] - want_crash) < 1e-6);
}

// --- Trainer ------------------------------------------------------------------------

GPU_TEST_CASE("racing ppo: training is bit-identical for the same seed") {
    RacingPpoConfig cfg;
    cfg.num_envs = 128;
    cfg.rollout_steps = 16;
    cfg.minibatches = 4;
    cfg.epochs = 2;
    std::vector<float> first_params, first_log_std;
    std::vector<float> first_curve;
    for (int run = 0; run < 2; ++run) {
        RacingPpoTrainer trainer(cfg);
        std::vector<float> curve;
        for (int u = 0; u < 4; ++u) {
            const auto s = trainer.update();
            curve.push_back(s.mean_laps);
            curve.push_back(s.approx_kl);
        }
        if (run == 0) {
            first_params  = trainer.actor_params();
            first_log_std = trainer.log_std();
            first_curve   = curve;
        } else {
            CHECK(trainer.actor_params() == first_params);
            CHECK(trainer.log_std() == first_log_std);
            CHECK(curve == first_curve);
        }
    }
}

GPU_TEST_CASE("racing ppo: the cars learn to drive on 3 seeds (laps rise, crashes fall)") {
    // Episodes end at the 50 s time limit (about four laps at top speed), so the
    // laps curve saturates there. Each seed should reach more than two and a half
    // laps on average, with most episodes ending on the time limit, not in a crash.
    for (uint32_t seed : {1u, 2u, 3u}) {
        RacingPpoConfig cfg;
        cfg.traffic = false; // original isolated circuit remains a learning regression
        cfg.seed = seed;
        cfg.total_updates = 100;
        RacingPpoTrainer trainer(cfg);
        float first_laps = 0.f, last_laps = 0.f, last_crash = 1.f;
        for (int u = 0; u < cfg.total_updates; ++u) {
            const auto s = trainer.update();
            if (u == 0) first_laps = s.mean_laps;
            last_laps  = s.mean_laps;
            last_crash = s.crash_fraction;
        }
        std::printf("    seed %u: laps %.2f -> %.2f, crashed %.1f%% of final episodes\n", seed, first_laps, last_laps,
                    100.0 * last_crash);
        const std::string tag = "seed " + std::to_string(seed) + ": ";
        CHECK_MSG(last_laps > 2.5f, tag + "laps after training " + std::to_string(last_laps));
        CHECK_MSG(last_crash < 0.1f, tag + "crash fraction " + std::to_string(last_crash));
        CHECK_MSG(last_laps > first_laps + 1.f, tag + "laps did not rise from " + std::to_string(first_laps));
    }
}
