#include "racing_showcase.h"

#include <stdexcept>

namespace cudarl::demos {

namespace {

constexpr int kCrashFlashFrames = 15;

nn::MlpSpec actor_spec(int hidden) {
    return nn::MlpSpec({racing::kObsDim, hidden, hidden, racing::kActionDim},
                       {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity});
}

} // namespace

RacingShowcase::RacingShowcase(int count, int hidden, uint32_t seed)
    : policy_(actor_spec(hidden)), seed_(seed), track_(racing::make_track()), states_(count), episodes_(count, 0),
      obs_(static_cast<size_t>(count) * racing::kObsDim, 0.f), crash_frames_(count, 0) {
    if (count <= 0) throw std::invalid_argument("RacingShowcase: count must be positive");
    reset();
}

void RacingShowcase::set_policy(const std::vector<float>& actor_params) {
    if (actor_params.size() != policy_.params().size()) throw std::invalid_argument("RacingShowcase: policy size");
    policy_.params() = actor_params;
}

void RacingShowcase::reset() {
    for (int i = 0; i < count(); ++i) {
        episodes_[i]     = 0;
        states_[i]       = racing::initial_state(seed_, static_cast<uint32_t>(i), 0);
        crash_frames_[i] = 0;
        observe(i);
    }
}

void RacingShowcase::observe(int i) {
    racing::write_obs(states_[i], track_, &obs_[static_cast<size_t>(i) * racing::kObsDim]);
}

void RacingShowcase::step() {
    const int n = count();
    policy_.forward(obs_.data(), n);
    const float* mean = policy_.output();
    for (int i = 0; i < n; ++i) {
        // The policy's mean action; racing::step clamps it to the valid range.
        const float action[racing::kActionDim] = {mean[i * racing::kActionDim + 0], mean[i * racing::kActionDim + 1]};
        float       final_obs[racing::kObsDim];
        const auto  r = racing::step(states_[i], action, track_, seed_, static_cast<uint32_t>(i), episodes_[i],
                                     final_obs);
        if (r.terminated) crash_frames_[i] = kCrashFlashFrames;
        else if (crash_frames_[i] > 0) --crash_frames_[i];
        observe(i);
    }
}

} // namespace cudarl::demos
