#ifndef CUDARL_DEMOS_CARTPOLE_SHOWCASE_H
#define CUDARL_DEMOS_CARTPOLE_SHOWCASE_H

#include <cstdint>
#include <vector>

#include "envs/cartpole/cartpole.h"
#include "nn/mlp_cpu.h"

// A handful of CartPoles run in real time for display, driven by a copy of the
// policy being trained. They step on the CPU (a dozen environments is nothing),
// so the showcase never competes with training for the GPU.
namespace cudarl::demos {

class CartPoleShowcase {
public:
    CartPoleShowcase(int count, int hidden, uint32_t seed);

    int count() const { return static_cast<int>(states_.size()); }
    void set_policy(const std::vector<float>& actor_params);
    void reset();
    // One 20 ms physics step for every cart. greedy: take the most likely action.
    void step(bool greedy);

    const cartpole::State& state(int i) const { return states_[i]; }
    int last_length(int i) const { return last_length_[i]; }      // length of the cart's previous episode
    bool just_failed(int i) const { return failed_frames_[i] > 0; } // the pole fell within the last few frames

private:
    nn::CpuMlp<float>             policy_;
    uint32_t                      seed_;
    uint64_t                      step_ = 0;
    std::vector<cartpole::State>  states_;
    std::vector<uint32_t>         episodes_;
    std::vector<int>              last_length_;
    std::vector<int>              failed_frames_;
};

} // namespace cudarl::demos

#endif // CUDARL_DEMOS_CARTPOLE_SHOWCASE_H
