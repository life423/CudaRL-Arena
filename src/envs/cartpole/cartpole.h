#ifndef CUDARL_ENVS_CARTPOLE_H
#define CUDARL_ENVS_CARTPOLE_H

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/hd.h"
#include "core/rng.h"
#include "cuda/cuda_memory.h"

// CartPole-v1 (Barto, Sutton & Anderson 1983, with the constants and Euler
// integration of the standard Gym/Gymnasium version). A pole is hinged to a
// cart on a track; push the cart left or right to keep the pole upright.
// Reward +1 per step; the episode terminates when the pole passes 12 degrees
// or the cart leaves +-2.4, and is truncated after 500 steps.
namespace cudarl::cartpole {

constexpr int   kObsDim       = 4;   // x, x_dot, theta, theta_dot
constexpr int   kNumActions   = 2;   // 0 push left, 1 push right
constexpr int   kMaxSteps     = 500;
constexpr float kGravity      = 9.8f;
constexpr float kMassCart     = 1.0f;
constexpr float kMassPole     = 0.1f;
constexpr float kTotalMass    = kMassCart + kMassPole;
constexpr float kHalfLength   = 0.5f;   // half the pole's length
constexpr float kPoleMassLen  = kMassPole * kHalfLength;
constexpr float kForceMag     = 10.0f;
constexpr float kTau          = 0.02f;  // seconds per step
constexpr float kThetaLimit   = 12.0f * 2.0f * 3.14159265358979f / 360.0f;
constexpr float kXLimit       = 2.4f;

struct State {
    float x, x_dot, theta, theta_dot;
    int   steps;   // steps taken in the current episode
};

// One Euler step of the cart-pole dynamics.
CUDARL_HD inline void integrate(State& s, int action) {
    const float force    = action == 1 ? kForceMag : -kForceMag;
    const float cos_th   = cosf(s.theta);
    const float sin_th   = sinf(s.theta);
    const float temp     = (force + kPoleMassLen * s.theta_dot * s.theta_dot * sin_th) / kTotalMass;
    const float theta_acc =
        (kGravity * sin_th - cos_th * temp) / (kHalfLength * (4.0f / 3.0f - kMassPole * cos_th * cos_th / kTotalMass));
    const float x_acc = temp - kPoleMassLen * theta_acc * cos_th / kTotalMass;
    s.x         += kTau * s.x_dot;
    s.x_dot     += kTau * x_acc;
    s.theta     += kTau * s.theta_dot;
    s.theta_dot += kTau * theta_acc;
}

CUDARL_HD inline bool terminated(const State& s) {
    return s.x < -kXLimit || s.x > kXLimit || s.theta < -kThetaLimit || s.theta > kThetaLimit;
}

// Start state uniform in [-0.05, 0.05]^4; a pure function of (seed, env, episode).
CUDARL_HD inline State initial_state(uint32_t seed, uint32_t env, uint32_t episode) {
    auto u = [&](uint32_t i) { return rng::uniform(seed, env, episode * 4u + i) * 0.1f - 0.05f; };
    return State{u(0), u(1), u(2), u(3), 0};
}

CUDARL_HD inline void write_obs(const State& s, float* obs) {
    obs[0] = s.x;
    obs[1] = s.x_dot;
    obs[2] = s.theta;
    obs[3] = s.theta_dot;
}

struct StepResult {
    float reward;
    bool  terminated;   // the pole fell or the cart left the track
    bool  truncated;    // hit the step limit while still balancing
};

// Advances one environment. When the episode ends, the caller gets the final
// (pre-reset) observation in final_obs and the state restarts as episode + 1.
CUDARL_HD inline StepResult step(State& s, int action, uint32_t seed, uint32_t env, uint32_t& episode,
                                 float* final_obs) {
    integrate(s, action);
    s.steps += 1;
    StepResult r{1.0f, terminated(s), false};
    r.truncated = !r.terminated && s.steps >= kMaxSteps;
    if (r.terminated || r.truncated) {
        write_obs(s, final_obs);
        episode += 1;
        s = initial_state(seed, env, episode);
    }
    return r;
}

// Per-step outputs of a batch, each [num_envs] (final_obs is [num_envs x 4]).
struct StepOutputs {
    float* reward;
    float* done;             // 1 if the episode ended this step (terminated or truncated)
    float* truncated;        // 1 if it ended by the step limit
    float* final_obs;        // the observation the episode ended on (valid where done)
    float* finished_length;  // the finished episode's length where done, else 0
};

// num_envs CartPoles stepped in parallel, one GPU thread each.
class CartPoleBatch {
public:
    CartPoleBatch(int num_envs, uint32_t seed);

    int num_envs() const { return n_; }
    void reset(cudaStream_t stream = nullptr);
    // Takes one action per environment; obs() then holds the next observations
    // (already reset for environments whose episode just ended).
    void step(const int* d_actions, const StepOutputs& out, cudaStream_t stream = nullptr);
    const float* obs() const { return obs_.get(); }   // device [num_envs x 4]
    const State* states() const { return states_.get(); }

private:
    int                  n_;
    uint32_t             seed_;
    device_ptr<State>    states_;
    device_ptr<uint32_t> episodes_;
    device_ptr<float>    obs_;
};

} // namespace cudarl::cartpole

#endif // CUDARL_ENVS_CARTPOLE_H
