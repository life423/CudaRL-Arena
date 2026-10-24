// GpuMlp vs. the double-precision CPU reference (outputs, parameter gradients,
// input gradients), plus run-to-run determinism.

#include <vector>

#include "check.h"
#include "gpu_util.h"
#include "nn/mlp_cpu.h"
#include "nn/mlp_gpu.h"

using namespace cudarl::nn;
using cudarl::test::random_floats;
using cudarl::test::to_device;
using cudarl::test::to_host;

namespace {

struct Result {
    std::vector<float> output, grads, dx;
};

Result run_gpu(const MlpSpec& spec, const std::vector<float>& params, const std::vector<float>& x,
               const std::vector<float>& dy, int batch, GemmKernel kernel) {
    GpuMlp net(spec, batch, 0, kernel);
    net.set_params(params);
    auto dx_in = to_device(x), d_dy = to_device(dy);
    device_ptr<float> d_dx(x.size());
    net.forward(dx_in.get(), batch);
    net.zero_grad();
    net.backward(d_dy.get(), batch, d_dx.get());
    return {to_host(net.output(), dy.size()), net.get_grads(), to_host(d_dx, x.size())};
}

// The real initializer (weights at 1/sqrt(fan_in)) plus small noise, so biases
// are non-zero too. Uniformly large weights would push pre-activations into a
// regime (|z| ~ 40) no trained network sits in, where fp32 rounding dominates.
std::vector<float> realistic_params(const MlpSpec& spec, uint32_t seed) {
    auto p = init_params(spec, seed);
    const auto noise = random_floats(p.size(), seed + 1000, 0.05f);
    for (size_t i = 0; i < p.size(); ++i) p[i] += noise[i];
    return p;
}

void compare_with_cpu(const MlpSpec& spec, int batch, uint32_t seed, GemmKernel kernel = GemmKernel::Tiled) {
    const auto params = realistic_params(spec, seed);
    const auto x  = random_floats(static_cast<size_t>(batch) * spec.input_size(), seed + 1);
    const auto dy = random_floats(static_cast<size_t>(batch) * spec.output_size(), seed + 2);

    CpuMlp<double> ref(spec);
    ref.params().assign(params.begin(), params.end());
    const std::vector<double> xd(x.begin(), x.end()), dyd(dy.begin(), dy.end());
    std::vector<double> ref_dx(x.size());
    ref.forward(xd.data(), batch);
    ref.zero_grad();
    ref.backward(dyd.data(), batch, ref_dx.data());
    const std::vector<double> ref_out(ref.output(), ref.output() + dy.size());

    const Result gpu = run_gpu(spec, params, x, dy, batch, kernel);
    CHECK_CLOSE(gpu.output, ref_out, 1e-5);
    CHECK_CLOSE(gpu.grads, ref.grads(), 1e-4);   // sums over the whole batch
    CHECK_CLOSE(gpu.dx, ref_dx, 1e-5);
}

} // namespace

GPU_TEST_CASE("gpu_mlp: matches CPU, single unit") {
    compare_with_cpu(MlpSpec({1, 1}, {Activation::Identity}), 1, 201);
}

GPU_TEST_CASE("gpu_mlp: matches CPU, odd sizes, mixed activations") {
    compare_with_cpu(MlpSpec({7, 13, 5, 3}, {Activation::Tanh, Activation::ReLU, Activation::Identity}), 37, 211);
}

GPU_TEST_CASE("gpu_mlp: matches CPU, PPO-sized policy at batch 4096") {
    compare_with_cpu(MlpSpec({8, 64, 64, 2}, {Activation::Tanh, Activation::Tanh, Activation::Identity}), 4096, 221);
}

GPU_TEST_CASE("gpu_mlp: matches CPU, wide layers spanning several tiles") {
    compare_with_cpu(MlpSpec({130, 200, 70}, {Activation::ReLU, Activation::Tanh}), 300, 231);
}

GPU_TEST_CASE("gpu_mlp: naive kernel gives the same answers") {
    compare_with_cpu(MlpSpec({7, 13, 5, 3}, {Activation::Tanh, Activation::ReLU, Activation::Identity}), 37, 241,
                     GemmKernel::Naive);
}

GPU_TEST_CASE("gpu_mlp: bit-identical across runs and instances") {
    const MlpSpec spec({8, 64, 64, 2}, {Activation::Tanh, Activation::Tanh, Activation::Identity});
    const int batch = 20000;   // large enough that dW uses split-K
    const auto params = realistic_params(spec, 301);
    const auto x  = random_floats(static_cast<size_t>(batch) * 8, 302);
    const auto dy = random_floats(static_cast<size_t>(batch) * 2, 303);

    const Result first = run_gpu(spec, params, x, dy, batch, GemmKernel::Tiled);
    for (int run = 0; run < 3; ++run) {
        const Result again = run_gpu(spec, params, x, dy, batch, GemmKernel::Tiled);
        CHECK(again.output == first.output);
        CHECK(again.grads == first.grads);
        CHECK(again.dx == first.dx);
    }
}

GPU_TEST_CASE("gpu_mlp: gradients accumulate until zero_grad") {
    const MlpSpec spec({4, 6, 3}, {Activation::Tanh, Activation::Identity});
    GpuMlp net(spec, 16, 5);
    auto x = to_device(random_floats(16 * 4, 311));
    auto dy = to_device(random_floats(16 * 3, 312));
    net.forward(x.get(), 16);
    net.zero_grad();
    net.backward(dy.get(), 16);
    const auto once = net.get_grads();
    net.backward(dy.get(), 16);
    const auto twice = net.get_grads();
    std::vector<float> doubled(once.size());
    for (size_t i = 0; i < once.size(); ++i) doubled[i] = 2.f * once[i];
    CHECK_CLOSE(twice, doubled, 1e-6);
}

GPU_TEST_CASE("gpu_mlp: rejects bad batch sizes") {
    GpuMlp net(MlpSpec({2, 2}, {Activation::Identity}), 8, 1);
    auto x = to_device(random_floats(32, 1));
    bool threw = false;
    try { net.forward(x.get(), 9); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    net.forward(x.get(), 4);
    try { net.backward(x.get(), 5); } catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
}
