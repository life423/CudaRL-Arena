// Verifies the CPU reference's analytic gradients against central finite
// differences in double precision. Everything else is tested against this
// reference, so it has to be right first.

#include <cmath>
#include <vector>

#include "check.h"
#include "core/rng.h"
#include "nn/mlp_cpu.h"

using namespace cudarl::nn;

namespace {

std::vector<double> random_vector(size_t n, uint32_t seed, double scale = 1.0) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * cudarl::rng::normal(seed, 0, static_cast<uint32_t>(i));
    return v;
}

// L = sum_i w_i * y_i, so dL/dy = w.
double loss(CpuMlp<double>& net, const std::vector<double>& x, int batch, const std::vector<double>& w) {
    net.forward(x.data(), batch);
    const double* y = net.output();
    double l = 0.0;
    for (size_t i = 0; i < w.size(); ++i) l += w[i] * y[i];
    return l;
}

// Central differences with h = 1e-5 balance truncation error (O(h^2)) against
// floating-point roundoff in the loss (O(eps * |L| / h)). Errors are compared
// over the whole gradient vector (max |analytic - numeric| / max |numeric|), since
// per-element relative error is meaningless for gradients near zero.
constexpr double kStep = 1e-5;
constexpr double kTolerance = 1e-7;

void gradient_check(const MlpSpec& spec, int batch, uint32_t seed) {
    CpuMlp<double> net(spec);
    // Non-zero biases too, so their gradients are exercised.
    net.params() = random_vector(spec.num_params(), seed, 0.5);
    const auto x = random_vector(static_cast<size_t>(batch) * spec.input_size(), seed + 1);
    const auto w = random_vector(static_cast<size_t>(batch) * spec.output_size(), seed + 2);

    loss(net, x, batch, w);
    net.zero_grad();
    std::vector<double> dx(x.size());
    net.backward(w.data(), batch, dx.data());
    const std::vector<double> analytic = net.grads();

    std::vector<double> numeric(analytic.size());
    for (int i = 0; i < spec.num_params(); ++i) {
        const double saved = net.params()[i];
        net.params()[i] = saved + kStep;
        const double up = loss(net, x, batch, w);
        net.params()[i] = saved - kStep;
        const double down = loss(net, x, batch, w);
        net.params()[i] = saved;
        numeric[i] = (up - down) / (2 * kStep);
    }
    CHECK_CLOSE(analytic, numeric, kTolerance);

    std::vector<double> numeric_dx(x.size());
    auto xp = x;
    for (size_t i = 0; i < x.size(); ++i) {
        xp[i] = x[i] + kStep;
        const double up = loss(net, xp, batch, w);
        xp[i] = x[i] - kStep;
        const double down = loss(net, xp, batch, w);
        xp[i] = x[i];
        numeric_dx[i] = (up - down) / (2 * kStep);
    }
    CHECK_CLOSE(dx, numeric_dx, kTolerance);
}

} // namespace

TEST_CASE("cpu_mlp: gradient check, single unit") {
    gradient_check(MlpSpec({1, 1}, {Activation::Identity}), 1, 11);
}

TEST_CASE("cpu_mlp: gradient check, tanh hidden layer") {
    gradient_check(MlpSpec({3, 5, 2}, {Activation::Tanh, Activation::Identity}), 4, 21);
}

TEST_CASE("cpu_mlp: gradient check, mixed activations, odd sizes") {
    gradient_check(MlpSpec({7, 4, 6, 3}, {Activation::ReLU, Activation::Tanh, Activation::Identity}), 5, 31);
}

TEST_CASE("cpu_mlp: gradient check, wider tanh network") {
    gradient_check(MlpSpec({33, 17, 9}, {Activation::Tanh, Activation::Tanh}), 3, 41);
}

TEST_CASE("cpu_mlp: gradients accumulate across backward calls") {
    const MlpSpec spec({4, 3, 2}, {Activation::Tanh, Activation::Identity});
    CpuMlp<double> net(spec);
    net.params() = random_vector(spec.num_params(), 51, 0.5);
    const auto x  = random_vector(8, 52);
    const auto dy = random_vector(4, 53);

    net.forward(x.data(), 2);
    net.zero_grad();
    net.backward(dy.data(), 2);
    const auto once = net.grads();
    net.backward(dy.data(), 2);
    for (size_t i = 0; i < once.size(); ++i) CHECK(std::fabs(net.grads()[i] - 2 * once[i]) < 1e-12);
}

TEST_CASE("cpu_mlp: parameter layout and init") {
    const MlpSpec spec({3, 5, 2}, {Activation::Tanh, Activation::Identity});
    CHECK(spec.num_params() == (5 * 3 + 5) + (2 * 5 + 2));
    CHECK(spec.weight_offset(1) == 20);
    CHECK(spec.bias_offset(1) == 30);

    const auto p = init_params(spec, 7);
    CHECK(p == init_params(spec, 7));                  // deterministic
    CHECK(p != init_params(spec, 8));                  // seed matters
    for (int b = spec.bias_offset(0); b < spec.weight_offset(1); ++b) CHECK(p[b] == 0.f);
}
