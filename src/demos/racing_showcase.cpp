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
    : policy_(actor_spec(hidden)), seed_(seed), track_(racing::make_track(true)), states_(count), episodes_(count, 0),
      obs_(static_cast<size_t>(count) * racing::kObsDim, 0.f), crash_frames_(count, 0) {
    if (count != racing::kRaceCars) throw std::invalid_argument("RacingShowcase: requires eight cars");
    reset();
}

void RacingShowcase::set_policy(const std::vector<float>& actor_params) {
    if (actor_params.size() != policy_.params().size()) throw std::invalid_argument("RacingShowcase: policy size");
    policy_.params() = actor_params;
}

void RacingShowcase::reset() {
    racing::reset_race(states_.data(),episodes_.data(),track_,seed_,0);
    clean_laps_=contacts_=wall_crashes_=0;
    best_lap_=0.f;
    for (int i = 0; i < count(); ++i) {
        crash_frames_[i] = 0;
        observe(i);
    }
}

void RacingShowcase::observe(int i) {
    racing::observe_race(states_.data(),i,track_,&obs_[static_cast<size_t>(i)*racing::kObsDim]);
}

void RacingShowcase::step() {
    const int n = count();
    policy_.forward(obs_.data(), n);
    const float* mean = policy_.output();
    const auto previous=states_;
    racing::StepResult results[racing::kRaceCars];
    float final_obs[racing::kRaceCars*racing::kObsDim];
    for (int i = 0; i < n; ++i) {
        // The policy's mean action; racing::step clamps it to the valid range.
        const float action[racing::kActionDim] = {mean[i * racing::kActionDim + 0], mean[i * racing::kActionDim + 1]};
        results[i]=racing::advance_race(states_[i],action,track_);
    }
    racing::resolve_race(states_.data(),previous.data(),results,episodes_.data(),track_,seed_,0,final_obs);
    for (int i=0;i<n;++i) {
        const auto& r=results[i];
        if (r.contact) ++contacts_;
        if (r.terminated && !r.contact) ++wall_crashes_;
        if (r.clean_lap) {
            ++clean_laps_;
            if (best_lap_==0.f || r.lap_seconds<best_lap_) best_lap_=r.lap_seconds;
        }
        if (r.terminated) crash_frames_[i] = kCrashFlashFrames;
        else if (crash_frames_[i] > 0) --crash_frames_[i];
        observe(i);
    }
}

} // namespace cudarl::demos
