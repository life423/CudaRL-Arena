#include "check.h"
#include "gpu_util.h"
#include "nn/mlp_cpu.h"
#include "rl/ar_gaussian_math.h"
#include "rl/gaussian_kernels.h"
#include <cstdio>
#include <vector>
using namespace cudarl;
using namespace cudarl::rl;
using cudarl::test::to_device;
using cudarl::test::to_host;

TEST_CASE("ar_gaussian: recorded sequence rescoring shared network weight finite differences") {
    const nn::MlpSpec topology{{40,32,32,4}, {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};
    nn::CpuMlp<double> net(topology);
    const auto initialized=nn::orthogonal_params(topology,1247037241,{1.414213562f,1.414213562f,.01f});
    net.params().assign(initialized.begin(),initialized.end());
    constexpr int n = 5, d = 4;
    // Current rows followed by their own predecessors; row3 is a true episode reset.
    const bool valid[5]{false, true, true, false, true};
    double obs[2*n*40]{},raw[n*d]{},previous_raw[n*d]{};
    for(int i=0;i<n;++i){for(int j=0;j<40;++j){obs[i*40+j]=.2*sin(double(i*40+j));if(valid[i])obs[(n+i)*40+j]=obs[(i-1)*40+j];}for(int j=0;j<d;++j){raw[i*d+j]=.3*sin(double(i*d+j)+.5);if(valid[i])previous_raw[i*d+j]=raw[(i-1)*d+j];}}
    const double sigma[4]{-1,-1,-1,-1};
    for (double rho : {0., .8}) {
        net.forward(obs, 2 * n);
        double old[n];
        for (int i = 0; i < n; ++i)
            old[i] = ar_logp(net.output() + i * d, net.output() + (n + i) * d, previous_raw + i * d, sigma, d,
                             rho, valid[i], raw + i * d);
        const auto initial = net.params();
        for (size_t j = 0; j < initial.size(); ++j)
            net.params()[j] += .001 * cos(double(j));
        auto loss = [&]() {
            net.forward(obs, 2 * n);
            double result = 0;
            for (int i = 0; i < n; ++i) {
                double independent = 0;
                for (int j = 0; j < d; ++j) {
                    const double mu =
                        net.output()[i * d + j] +
                        (valid[i] ? rho * (previous_raw[i * d + j] - net.output()[(n + i) * d + j]) : 0);
                    const double variance = exp(-2.) * (valid[i] ? 1 - rho * rho : 1);
                    const double error = raw[i * d + j] - mu;
                    independent +=
                        -.5 * (error * error / variance + log(2 * 3.14159265358979323846 * variance));
                }
                const double actual = ar_logp(net.output() + i * d, net.output() + (n + i) * d,
                                              previous_raw + i * d, sigma, d, rho, valid[i], raw + i * d);
                CHECK(fabs(actual - independent) < 1e-12);
                result -= exp(actual - old[i]) / n;
            }
            return result;
        };
        (void)loss();
        double gradient[2*n*d]{};
        for (int i = 0; i < n; ++i) {
            double value_gradient;
            ppo_ar_row(net.output() + i * d, net.output() + (n + i) * d, previous_raw + i * d, sigma, d, rho,
                       valid[i], raw + i * d, old[i], 1., 0., 0., {.2f, 0, 0}, 1. / n, gradient + i * d,
                       gradient + (n + i) * d, &value_gradient);
        }
        net.zero_grad();
        net.backward(gradient, 2 * n);
        const auto analytic = net.grads();
        for (size_t j = 0; j < net.params().size(); ++j) {
            const double base = net.params()[j], h = 1e-6;
            net.params()[j] = base + h;
            const double up = loss();
            net.params()[j] = base - h;
            const double down = loss();
            net.params()[j] = base;
            CHECK(fabs(analytic[j] - (up - down) / (2 * h)) < 1e-7);
        }
        net.params() = initial;
    }
}

TEST_CASE("ar_gaussian: exact density ratios reset gradients and weight boundary") {
    double current[2]{.2, -.1}, previous[2]{-.1, .3}, past[2]{.5, -.2}, ls[2]{-1, -1}, action[2]{.3, -.4};
    for (double rho : {0., .8})
        for (bool history : {false, true}) {
            double mu[2], conditional[2], dc[2], dp[2], dv;
            ar_parameters(current, previous, past, ls, 2, rho, history, mu, conditional);
            const double old = ar_logp(current, previous, past, ls, 2, rho, history, action);
            CHECK(fabs(old - gaussian_logp(mu, conditional, action, 2)) < 1e-12);
            if (!history || rho == 0)
                CHECK(fabs(old - gaussian_logp(current, ls, action, 2)) < 1e-12);
            const auto terms = ppo_ar_row(current, previous, past, ls, 2, rho, history, action, old, 1., .2,
                                          .1, {.2f, 0, .5f}, 1., dc, dp, &dv);
            CHECK(fabs(terms.approx_kl) < 1e-12);
            if (rho == 0) {
                double iid_gradient[2], iid_sigma_gradient[2], iid_value;
                ppo_gaussian_row(current, ls, 2, action, old, 1., .2, .1, {.2f, 0, .5f}, 1.,
                                 iid_gradient, iid_sigma_gradient, &iid_value);
                for (int j = 0; j < 2; ++j) { CHECK(dc[j] == iid_gradient[j]); CHECK(dp[j] == 0); }
                CHECK(dv == iid_value);
            }
            for (int j = 0; j < 2; ++j)
                for (int which = 0; which < 2; ++which) {
                    double *field = which ? previous : current;
                    const double saved = field[j], h = 1e-6;
                    field[j] = saved + h;
                    const double up =
                        -exp(ar_logp(current, previous, past, ls, 2, rho, history, action) - old);
                    field[j] = saved - h;
                    const double down =
                        -exp(ar_logp(current, previous, past, ls, 2, rho, history, action) - old);
                    field[j] = saved;
                    CHECK(fabs((up - down) / (2 * h) - (which ? dp[j] : dc[j])) < 1e-7);
                }
            // A shared network bias moves BOTH means. Detaching the previous mean fails this.
            const double h = 1e-6;
            current[0] += h;
            previous[0] += h;
            const double up = -exp(ar_logp(current, previous, past, ls, 2, rho, history, action) - old);
            current[0] -= 2 * h;
            previous[0] -= 2 * h;
            const double down = -exp(ar_logp(current, previous, past, ls, 2, rho, history, action) - old);
            current[0] += h;
            previous[0] += h;
            CHECK(fabs((up - down) / (2 * h) - (dc[0] + dp[0])) < 1e-7);
            // Tanh density includes the Jacobian; its parameter-independent part cancels in ratios.
            const double jac = tanh_log_jacobian(action, 2);
            current[0] += .01;
            const double next = ar_logp(current, previous, past, ls, 2, rho, history, action);
            CHECK(fabs(exp(next - old) - exp((next - jac) - (old - jac))) < 1e-12);
            current[0] -= .01;
            // Recompute the previous mean after a policy version changes, rather than old epsilon.
            previous[0] += .1;
            double new_mu[2], new_ls[2];
            ar_parameters(current, previous, past, ls, 2, rho, history, new_mu, new_ls);
            CHECK(fabs(new_mu[0] - mu[0] + (history ? rho * .1 : 0)) < 1e-12);
            previous[0] -= .1;
        }
}

GPU_TEST_CASE("ar_gaussian: CUDA sampling and both mean gradients match host") {
    constexpr int n = 17, dim = 4;
    std::vector<float> means(2 * n * dim), past(n * dim), history(n), sigma(dim, -1), adv(n), values(n, .2f),
        returns(n, .1f);
    for (int i = 0; i < 2 * n * dim; ++i)
        means[i] = .2f * sinf(float(i));
    for (int i = 0; i < n * dim; ++i)
        past[i] = .3f * cosf(float(i));
    for (int i = 0; i < n; ++i) {
        history[i] = i % 3 ? 1.f : 0.f;
        adv[i] = i % 2 ? 1.f : -1.f;
    }
    auto dm = to_device(means), dp = to_device(past), dh = to_device(history), ds = to_device(sigma),
         da = to_device(adv), dv = to_device(values), dr = to_device(returns),
         stats_adv = to_device(std::vector<float>{0, 1});
    device_ptr<float> sampled(n * dim), old(n), gradient(2 * n * dim), vg(n), rows(n * 5), stats(5);
    for (float rho : {0.f, .8f}) {
        sample_ar_gaussian(dm.get(), ds.get(), dp.get(), dh.get(), n, dim, rho, 119, 7, sampled.get(),
                           old.get(), nullptr);
        const auto action = to_host(sampled, n * dim), lp = to_host(old, n);
        if (rho == 0) {
            device_ptr<float> iid_actions(n*dim), iid_logp(n);
            sample_gaussian(dm.get(),ds.get(),n,dim,119,7,iid_actions.get(),iid_logp.get(),nullptr);
            CHECK(action == to_host(iid_actions,n*dim)); CHECK(lp == to_host(iid_logp,n));
        }
        for (int i = 0; i < n; ++i)
            CHECK(fabsf(lp[i] - ar_logp(means.data() + i * dim, means.data() + (n + i) * dim,
                                        past.data() + i * dim, sigma.data(), dim, rho, history[i] != 0,
                                        action.data() + i * dim)) < 2e-5f);
        CUDA_CHECK_THROW(cudaMemset(stats.get(), 0, stats.size_bytes()));
        ar_gaussian_ppo_loss_grad(dm.get(), ds.get(), dp.get(), dh.get(), dv.get(), n, dim, rho,
                                  sampled.get(), old.get(), da.get(), dr.get(), stats_adv.get(),
                                  {.2f, 0, .5f}, gradient.get(), vg.get(), rows.get(), stats.get(), nullptr);
        const auto g = to_host(gradient, 2 * n * dim), v = to_host(vg, n);
        for (int i = 0; i < n; ++i) {
            float gc[4], gp[4], gv;
            ppo_ar_row(means.data() + i * dim, means.data() + (n + i) * dim, past.data() + i * dim,
                       sigma.data(), dim, rho, history[i] != 0, action.data() + i * dim, lp[i], adv[i],
                       values[i], returns[i], {.2f, 0, .5f}, 1.f / n, gc, gp, &gv);
            for (int j = 0; j < 4; ++j) {
                CHECK(fabsf(g[i * dim + j] - gc[j]) < 1e-5f);
                CHECK(fabsf(g[(n + i) * dim + j] - gp[j]) < 1e-5f);
            }
            CHECK(fabsf(v[i] - gv) < 1e-6f);
        }
    }
}

GPU_TEST_CASE("ar_gaussian: stationary marginal amplitude ACF and cross joint independence") {
    // Frozen tolerances: SD within2%, ACF within.015, cross correlation<.02,
    // matched decoded RMS within2% under the identical constant actor mean.
    constexpr int n = 2048, dim = 4, steps = 128;
    std::vector<float> means(2 * n * dim, .2f), sigma(dim, -1), zero(n * dim), hist(n, 1);
    auto dm = to_device(means), ds = to_device(sigma), dz = to_device(zero), dh = to_device(hist),
         reset = to_device(std::vector<float>(n, 0));
    double decoded_rms[2][4]{};
    for (int method = 0; method < 2; ++method) {
        const float rho = method ? .8f : 0.f;
        device_ptr<float> all(n * dim * steps), lp(n);
        for (int t = 0; t < steps; ++t)
            sample_ar_gaussian(dm.get(), ds.get(), t ? all.get() + (t - 1) * n * dim : dz.get(),
                               t ? dh.get() : reset.get(), n, dim, rho, 843, static_cast<uint32_t>(t),
                               all.get() + t * n * dim, lp.get(), nullptr);
        const auto raw = to_host(all, n * dim * steps);
        double sums[4]{}, sq[4]{}, lag[4]{}, decoded[4]{}, cross[4][4]{};
        for (int t = 0; t < steps; ++t)
            for (int i = 0; i < n; ++i) {
                double e[4];
                for (int j = 0; j < 4; ++j) {
                    const int at = (t * n + i) * dim + j;
                    e[j] = raw[at] - .2;
                    const double d = tanh(raw[at]) - tanh(.2);
                    sums[j] += e[j];
                    sq[j] += e[j] * e[j];
                    decoded[j] += d * d;
                    if (t)
                        lag[j] += e[j] * (raw[at - n * dim] - .2);
                }
                for (int j = 0; j < 4; ++j)
                    for (int k = j + 1; k < 4; ++k)
                        cross[j][k] += e[j] * e[k];
            }
        for (int j = 0; j < 4; ++j) {
            const double count = n * steps, mean = sums[j] / count, sd = sqrt(sq[j] / count - mean * mean),
                         acf = lag[j] / (n * (steps - 1)) / (sq[j] / count);
            decoded_rms[method][j] = sqrt(decoded[j] / count);
            std::printf("AR_NOISE rho=%.1f joint=%d latent_SD=%.8f lag1=%.8f decoded_RMS=%.8f\n", rho, j, sd,
                        acf, decoded_rms[method][j]);
            CHECK(fabs(sd / exp(-1.) - 1) < .02);
            CHECK(fabs(acf - rho) < .015);
            for (int k = j + 1; k < 4; ++k)
                CHECK(fabs(cross[j][k] / sqrt(sq[j] * sq[k])) < .02);
        }
    }
    for (int j = 0; j < 4; ++j)
        CHECK(fabs(decoded_rms[1][j] / decoded_rms[0][j] - 1) < .02);
}
