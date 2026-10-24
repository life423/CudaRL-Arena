// The live-training demo's trainer: every target is learnable, predictions
// line up with the training loss, and bad configs are rejected.

#include <cmath>
#include <string>
#include <vector>

#include "check.h"
#include "demos/function_fit.h"

using namespace cudarl::demos;

namespace {

FitConfig config_for(FitTarget target, cudarl::nn::Activation act = cudarl::nn::Activation::Tanh) {
    FitConfig c;
    c.target     = target;
    c.activation = act;
    return c;
}

} // namespace

GPU_TEST_CASE("function_fit: sine is learned to MSE < 1e-3") {
    FunctionFit fit(config_for(FitTarget::Sine));
    const FitStats s = fit.train(1500);
    CHECK(s.step == 1500);
    CHECK_MSG(s.loss < 1e-3f, "loss " + std::to_string(s.loss));
    CHECK(s.ms_per_step > 0.0);
}

GPU_TEST_CASE("function_fit: every target improves at least 10x") {
    for (int t = 0; t < kNumFitTargets; ++t) {
        FunctionFit fit(config_for(static_cast<FitTarget>(t)));
        const float before = fit.train(1).loss;
        const float after  = fit.train(1500).loss;
        CHECK_MSG(after < before / 10, std::string(FunctionFit::target_name(static_cast<FitTarget>(t))) + ": " +
                                           std::to_string(before) + " -> " + std::to_string(after));
    }
}

GPU_TEST_CASE("function_fit: ReLU network learns the V shape") {
    FunctionFit fit(config_for(FitTarget::Abs, cudarl::nn::Activation::ReLU));
    CHECK(fit.train(1500).loss < 1e-3f);
}

GPU_TEST_CASE("function_fit: predictions agree with the reported loss") {
    FunctionFit fit(config_for(FitTarget::Sine));
    const float loss = fit.train(1500).loss;
    std::vector<float> xs(200);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = FunctionFit::kXMin + (FunctionFit::kXMax - FunctionFit::kXMin) * i / (xs.size() - 1);
    }
    const auto ys = fit.predict(xs);
    CHECK(ys.size() == xs.size());
    double sq = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) sq += std::pow(ys[i] - std::sin(xs[i]), 2);
    CHECK(sq / xs.size() < 4 * loss + 1e-5);   // similar error on a different grid
}

TEST_CASE("function_fit: rejects invalid configs") {
    for (auto bad : {FitConfig{FitTarget::Sine, 0}, FitConfig{FitTarget::Sine, 32, 0},
                     FitConfig{FitTarget::Sine, 32, 9}}) {
        bool threw = false;
        try { FunctionFit fit(bad); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
}

TEST_CASE("function_fit: targets stay in a drawable range") {
    for (int t = 0; t < kNumFitTargets; ++t) {
        for (int i = 0; i <= 100; ++i) {
            const float x = FunctionFit::kXMin + (FunctionFit::kXMax - FunctionFit::kXMin) * i / 100.f;
            CHECK(std::fabs(FunctionFit::target_value(static_cast<FitTarget>(t), x)) <= 1.f);
        }
    }
}
