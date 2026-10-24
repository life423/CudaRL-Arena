#include "cartpole_showcase.h"

#include <stdexcept>

#include "rl/ppo_math.h"

namespace cudarl::demos {

namespace {

constexpr int kFailFlashFrames = 15;

nn::MlpSpec actor_spec(int hidden) {
    return nn::MlpSpec({cartpole::kObsDim, hidden, hidden, cartpole::kNumActions},
                       {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity});
}

} // namespace

CartPoleShowcase::CartPoleShowcase(int count, int hidden, uint32_t seed)
    : policy_(actor_spec(hidden)), seed_(seed), states_(count), episodes_(count, 0), last_length_(count, 0),
      failed_frames_(count, 0) {
    if (count <= 0) throw std::invalid_argument("CartPoleShowcase: count must be positive");
    reset();
}

void CartPoleShowcase::set_policy(const std::vector<float>& actor_params) {
    if (actor_params.size() != policy_.params().size()) throw std::invalid_argument("CartPoleShowcase: policy size");
    policy_.params() = actor_params;
}

void CartPoleShowcase::reset() {
    for (int i = 0; i < count(); ++i) {
        episodes_[i] = 0;
        states_[i] = cartpole::initial_state(seed_, static_cast<uint32_t>(i), 0);
        last_length_[i] = 0;
        failed_frames_[i] = 0;
    }
}

void CartPoleShowcase::step(bool greedy) {
    const int n = count();
    std::vector<float> obs(static_cast<size_t>(n) * cartpole::kObsDim);
    for (int i = 0; i < n; ++i) cartpole::write_obs(states_[i], &obs[i * cartpole::kObsDim]);
    policy_.forward(obs.data(), n);
    const float* logits = policy_.output();
    for (int i = 0; i < n; ++i) {
        float logp[cartpole::kNumActions];
        rl::log_softmax(logits + i * cartpole::kNumActions, cartpole::kNumActions, logp);
        const float u = rng::uniform(seed_ ^ 0x5eedu, static_cast<uint32_t>(i), static_cast<uint32_t>(step_));
        const int action = rl::sample_action(logp, cartpole::kNumActions, u, greedy);
        const int length = states_[i].steps + 1;
        float final_obs[cartpole::kObsDim];
        const auto r = cartpole::step(states_[i], action, seed_, static_cast<uint32_t>(i), episodes_[i], final_obs);
        if (r.terminated || r.truncated) last_length_[i] = length;
        if (r.terminated) failed_frames_[i] = kFailFlashFrames;
        else if (failed_frames_[i] > 0) --failed_frames_[i];
    }
    ++step_;
}

} // namespace cudarl::demos
