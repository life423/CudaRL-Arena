// GpuAdam vs. the CPU reference, gradient clipping, determinism, and an
// end-to-end check that the whole stack (GEMMs, backward, Adam) learns.

#include <cmath>
#include <vector>

#include "check.h"
#include "gpu_util.h"
#include "nn/adam.h"
#include "nn/loss.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"

using namespace cudarl::nn;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

namespace {

AdamConfig adam_config(float lr, float max_grad_norm = 0.f) {
    AdamConfig c;
    c.lr = lr;
    c.max_grad_norm = max_grad_norm;
    return c;
}

// Runs `steps` Adam updates on both backends with a fresh random gradient each step.
void compare_adam(AdamConfig cfg, float grad_scale, int n, int steps) {
    const auto p0 = random_floats(n, 400);
    auto d_params = to_device(p0);
    device_ptr<float> d_grads(n);
    GpuAdam gpu(n, cfg);

    std::vector<double> cpu_params(p0.begin(), p0.end());
    CpuAdam<double> cpu(n, cfg);

    for (int t = 0; t < steps; ++t) {
        const auto g = random_floats(n, 1000 + t, grad_scale);
        d_grads.copy_from_host(g.data(), g.size());
        gpu.step(d_params.get(), d_grads.get());
        cpu.step(cpu_params, std::vector<double>(g.begin(), g.end()));
    }
    CHECK_CLOSE(to_host(d_params, n), cpu_params, 1e-5);
}

double sin_fit_mse(GpuMlp& net, GpuAdam& opt, const device_ptr<float>& x, const device_ptr<float>& target,
                   device_ptr<float>& dy, int batch, int steps) {
    for (int s = 0; s < steps; ++s) {
        net.forward(x.get(), batch);
        mse_loss_grad(net.output(), target.get(), dy.get(), batch);
        net.zero_grad();
        net.backward(dy.get(), batch);
        opt.step(net.params(), net.grads());
    }
    net.forward(x.get(), batch);
    const auto y = to_host(net.output(), batch);
    const auto t = to_host(target, batch);
    double mse = 0.0;
    for (int i = 0; i < batch; ++i) mse += (y[i] - t[i]) * (y[i] - t[i]);
    return mse / batch;
}

} // namespace

GPU_TEST_CASE("adam: matches CPU reference over 200 steps") {
    compare_adam(adam_config(1e-2f), 1.f, 1000, 200);
}

GPU_TEST_CASE("adam: clipping matches CPU when it triggers") {
    compare_adam(adam_config(1e-2f, 1.f), 10.f, 5000, 50);
}

GPU_TEST_CASE("adam: clipping is a no-op below the threshold") {
    // ||g|| ~ 0.01 * sqrt(1000) ~ 0.3, far below 100.
    compare_adam(adam_config(1e-2f, 100.f), 0.01f, 1000, 50);
}

GPU_TEST_CASE("adam: reports the pre-clip gradient norm") {
    const int n = 100000;
    const auto g = random_floats(n, 77, 3.f);
    double sq = 0.0;
    for (float v : g) sq += static_cast<double>(v) * v;
    auto d_params = to_device(std::vector<float>(n, 0.f));
    auto d_grads  = to_device(g);
    GpuAdam opt(n, adam_config(3e-4f, 1.f));
    opt.step(d_params.get(), d_grads.get());
    CHECK(std::fabs(opt.last_grad_norm() - std::sqrt(sq)) / std::sqrt(sq) < 1e-5);
}

GPU_TEST_CASE("adam: fits sin(x) end to end on the GPU") {
    const MlpSpec spec({1, 32, 32, 1}, {Activation::Tanh, Activation::Tanh, Activation::Identity});
    const int batch = 256;
    std::vector<float> xs(batch), ts(batch);
    for (int i = 0; i < batch; ++i) {
        xs[i] = -3.14159265f + 6.2831853f * i / (batch - 1);
        ts[i] = std::sin(xs[i]);
    }
    auto x = to_device(xs), target = to_device(ts);
    device_ptr<float> dy(batch);

    GpuMlp net(spec, batch, 1);
    GpuAdam opt(net.num_params(), adam_config(1e-2f, 1.f));
    const double before = sin_fit_mse(net, opt, x, target, dy, batch, 0);
    const double after  = sin_fit_mse(net, opt, x, target, dy, batch, 2000);
    CHECK_MSG(after < 1e-3, "final MSE " + std::to_string(after));
    CHECK_MSG(after < before / 100, "MSE only went from " + std::to_string(before) + " to " + std::to_string(after));
}

GPU_TEST_CASE("adam: training is bit-identical run to run") {
    const MlpSpec spec({1, 32, 32, 1}, {Activation::Tanh, Activation::Tanh, Activation::Identity});
    const int batch = 256;
    auto x = to_device(random_floats(batch, 5)), target = to_device(random_floats(batch, 6));
    device_ptr<float> dy(batch);
    std::vector<float> first;
    for (int run = 0; run < 2; ++run) {
        GpuMlp net(spec, batch, 1);
        GpuAdam opt(net.num_params(), adam_config(1e-2f, 0.5f));
        sin_fit_mse(net, opt, x, target, dy, batch, 100);
        const auto p = net.get_params();
        if (run == 0) first = p;
        else CHECK(p == first);
    }
}
