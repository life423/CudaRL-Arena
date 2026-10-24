#ifndef CUDARL_DEMOS_FUNCTION_FIT_H
#define CUDARL_DEMOS_FUNCTION_FIT_H

#include <cstdint>
#include <memory>
#include <vector>

#include "cuda/cuda_memory.h"
#include "nn/adam.h"
#include "nn/mlp_gpu.h"

// A small live-training demo: an MLP learns a 1-D function y = f(x) on
// [-pi, pi] by regression, entirely on the GPU. Godot calls train() every frame
// and draws predict() so you can watch the network's curve bend into shape.
namespace cudarl::demos {

enum class FitTarget : int { Sine = 0, Wiggle = 1, Square = 2, Abs = 3, Bump = 4 };
constexpr int kNumFitTargets = 5;

struct FitConfig {
    FitTarget      target     = FitTarget::Sine;
    int            width      = 32;    // hidden units per layer
    int            depth      = 2;     // hidden layers
    nn::Activation activation = nn::Activation::Tanh;
    float          lr         = 1e-2f;
    int            batch      = 1024;  // training points, evenly spaced over the domain
    uint32_t       seed       = 1;
};

struct FitStats {
    long long step        = 0;     // total optimizer steps so far
    float     loss        = 0.f;   // MSE over the training points after the last step
    double    ms_per_step = 0.0;   // GPU time per step in the last train() call
};

class FunctionFit {
public:
    static constexpr float kXMin = -3.14159265f;
    static constexpr float kXMax = 3.14159265f;

    explicit FunctionFit(const FitConfig& config);

    const FitConfig& config() const { return cfg_; }
    int num_params() const { return net_->num_params(); }

    // Runs `steps` full-batch Adam steps (MSE loss) and reports the result. Blocks
    // until the GPU finishes, so the returned loss is current.
    FitStats train(int steps);

    // Network outputs at arbitrary x values (at most config().batch of them).
    std::vector<float> predict(const std::vector<float>& xs);

    static float       target_value(FitTarget target, float x);
    static const char* target_name(FitTarget target);

private:
    FitConfig                    cfg_;
    std::unique_ptr<nn::GpuMlp>  net_;
    std::unique_ptr<nn::GpuAdam> opt_;
    device_ptr<float>            x_, target_, dy_, query_;
    std::vector<float>           targets_;   // host copy, for the loss
    long long                    step_ = 0;
    float                        loss_ = 0.f;
};

} // namespace cudarl::demos

#endif // CUDARL_DEMOS_FUNCTION_FIT_H
