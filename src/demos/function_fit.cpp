#include "function_fit.h"

#include <cmath>
#include <stdexcept>

#include "nn/loss.h"

namespace cudarl::demos {

namespace {

nn::MlpSpec make_spec(const FitConfig& c) {
    std::vector<int>            sizes{1};
    std::vector<nn::Activation> acts;
    for (int i = 0; i < c.depth; ++i) {
        sizes.push_back(c.width);
        acts.push_back(c.activation);
    }
    sizes.push_back(1);
    acts.push_back(nn::Activation::Identity);
    return nn::MlpSpec(sizes, acts);
}

nn::AdamConfig make_adam(const FitConfig& c) {
    nn::AdamConfig a;
    a.lr            = c.lr;
    a.max_grad_norm = 1.f;
    return a;
}

} // namespace

FunctionFit::FunctionFit(const FitConfig& config) : cfg_(config) {
    if (cfg_.width < 1 || cfg_.width > 1024 || cfg_.depth < 1 || cfg_.depth > 8 || cfg_.batch < 2) {
        throw std::invalid_argument("FunctionFit: width in [1, 1024], depth in [1, 8], batch >= 2");
    }
    net_ = std::make_unique<nn::GpuMlp>(make_spec(cfg_), cfg_.batch, cfg_.seed);
    opt_ = std::make_unique<nn::GpuAdam>(net_->num_params(), make_adam(cfg_));

    std::vector<float> xs(cfg_.batch);
    targets_.resize(cfg_.batch);
    for (int i = 0; i < cfg_.batch; ++i) {
        xs[i]       = kXMin + (kXMax - kXMin) * static_cast<float>(i) / static_cast<float>(cfg_.batch - 1);
        targets_[i] = target_value(cfg_.target, xs[i]);
    }
    x_.allocate(cfg_.batch);
    target_.allocate(cfg_.batch);
    dy_.allocate(cfg_.batch);
    query_.allocate(cfg_.batch);
    x_.copy_from_host(xs.data(), xs.size());
    target_.copy_from_host(targets_.data(), targets_.size());
}

FitStats FunctionFit::train(int steps) {
    FitStats stats;
    if (steps > 0) {
        cudaEvent_t start, stop;
        CUDA_CHECK_THROW(cudaEventCreate(&start));
        CUDA_CHECK_THROW(cudaEventCreate(&stop));
        CUDA_CHECK_THROW(cudaEventRecord(start));
        for (int s = 0; s < steps; ++s) {
            net_->forward(x_.get(), cfg_.batch);
            nn::mse_loss_grad(net_->output(), target_.get(), dy_.get(), cfg_.batch);
            net_->zero_grad();
            net_->backward(dy_.get(), cfg_.batch);
            opt_->step(net_->params(), net_->grads());
        }
        CUDA_CHECK_THROW(cudaEventRecord(stop));
        CUDA_CHECK_THROW(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start, stop));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        step_ += steps;
        stats.ms_per_step = ms / steps;

        // Loss of the updated network, from one more forward pass over the training points.
        net_->forward(x_.get(), cfg_.batch);
        std::vector<float> p(cfg_.batch);
        CUDA_CHECK_THROW(cudaMemcpy(p.data(), net_->output(), p.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double sq = 0.0;
        for (int i = 0; i < cfg_.batch; ++i) sq += static_cast<double>(p[i] - targets_[i]) * (p[i] - targets_[i]);
        loss_ = static_cast<float>(sq / cfg_.batch);
    }
    stats.step = step_;
    stats.loss = loss_;
    return stats;
}

std::vector<float> FunctionFit::predict(const std::vector<float>& xs) {
    if (xs.empty()) return {};
    if (static_cast<int>(xs.size()) > cfg_.batch) throw std::invalid_argument("FunctionFit::predict: too many points");
    query_.copy_from_host(xs.data(), xs.size());
    net_->forward(query_.get(), static_cast<int>(xs.size()));
    std::vector<float> out(xs.size());
    CUDA_CHECK_THROW(cudaMemcpy(out.data(), net_->output(), xs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

float FunctionFit::target_value(FitTarget target, float x) {
    switch (target) {
        case FitTarget::Wiggle: return std::sin(3.f * x) * (0.5f + 0.15f * x);
        case FitTarget::Square: return std::sin(x) >= 0.f ? 0.8f : -0.8f;
        case FitTarget::Abs:    return std::fabs(x) / 3.14159265f * 1.6f - 0.8f;
        case FitTarget::Bump:   return 1.6f * std::exp(-2.f * x * x) - 0.8f;
        default:                return std::sin(x);
    }
}

const char* FunctionFit::target_name(FitTarget target) {
    switch (target) {
        case FitTarget::Wiggle: return "Wiggle: sin(3x)(0.5 + 0.15x)";
        case FitTarget::Square: return "Square wave";
        case FitTarget::Abs:    return "V shape: |x|";
        case FitTarget::Bump:   return "Gaussian bump";
        default:                return "Sine: sin(x)";
    }
}

} // namespace cudarl::demos
