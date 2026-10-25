#ifndef CUDARL_DEMOS_RACING_SHOWCASE_H
#define CUDARL_DEMOS_RACING_SHOWCASE_H

#include <cstdint>
#include <vector>

#include "envs/racing/racing.h"
#include "nn/mlp_cpu.h"

// A handful of cars driven in real time by a copy of the policy being trained.
// They step on the CPU with the same racing:: code the GPU batch runs, so the
// showcase never competes with training for the GPU. Cars drive the policy's
// mean action (no sampling noise), which is how a trained policy is judged.
namespace cudarl::demos {

class RacingShowcase {
public:
    RacingShowcase(int count, int hidden, uint32_t seed);

    int count() const { return static_cast<int>(states_.size()); }
    void set_policy(const std::vector<float>& actor_params);
    void reset();
    // One physics step (racing::kDt) for every car.
    void step();

    const racing::State& state(int i) const { return states_[i]; }
    // The car's raycast distances in metres, kRays of them.
    float ray(int i, int r) const { return obs_[i * racing::kObsDim + r] * racing::kRayRange; }
    bool  just_crashed(int i) const { return crash_frames_[i] > 0; }     // the car hit a wall within the last few frames
    const racing::Track& track() const { return track_; }

private:
    void observe(int i);

    nn::CpuMlp<float>          policy_;
    uint32_t                   seed_;
    racing::Track              track_;
    std::vector<racing::State> states_;
    std::vector<uint32_t>      episodes_;
    std::vector<float>         obs_;       // [count x kObsDim], for the policy and for drawing the rays
    std::vector<int>           crash_frames_;
};

} // namespace cudarl::demos

#endif // CUDARL_DEMOS_RACING_SHOWCASE_H
