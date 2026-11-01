#ifndef CUDARL_RL_PPO_CHECKPOINT_H
#define CUDARL_RL_PPO_CHECKPOINT_H
#include "core/checkpoint.h"
#include "nn/adam_config.h"
#include "nn/motor/observation_contract.h"
#include <cmath>
#include <optional>
namespace cudarl::rl {
struct FrozenPpoActor {
    uint32_t seed = 0;
    uint32_t reward_version = 4;
    uint32_t observation_schema = motor_observation::legacy;
    float correlation = 0;
    int update = 0, actor_steps = 0, critic_steps = 0;
    std::vector<float> parameters, log_sigma;
};
struct PpoAdamState {
    nn::AdamConfig config{};
    int steps = 0;
    std::vector<float> first, second;
};
struct PpoTrainingState {
    FrozenPpoActor actor;
    std::vector<float> critic;
    PpoAdamState actor_adam, critic_adam;
    uint64_t source_protocol = 0;
};
inline std::vector<uint8_t> adam_payload(const PpoAdamState &state) {
    checkpoint::Writer out;
    out.value(state.config);
    out.value(state.steps);
    out.array(state.first);
    out.array(state.second);
    return out.data;
}
// PPO research schema2 fixes topology40-32-32-(4actor/1critic), float32,
// model6/task6/reward4 or5 and independent joint-target mapping3.
// Parse/validate auxiliary state without constructing or restoring an optimizer.
inline PpoTrainingState
read_ppo_training_state(const std::vector<uint8_t> &payload, uint64_t expected_protocol,
                        std::optional<uint32_t> expected_seed = std::nullopt,
                        std::optional<uint32_t> expected_observations = std::nullopt) {
    checkpoint::Reader in{payload};
    if (in.value<uint32_t>() != 0x31525050)
        throw std::runtime_error("Wrong PPO research schema");
    const auto schema = in.value<uint32_t>();
    if (schema != 2 && schema != 3 && schema != 4)
        throw std::runtime_error("Wrong PPO research schema");
    PpoTrainingState state;
    auto &actor = state.actor;
    actor.seed = in.value<uint32_t>();
    state.source_protocol = in.value<uint64_t>();
    if ((expected_seed && actor.seed != *expected_seed) || state.source_protocol != expected_protocol)
        throw std::runtime_error("PPO source protocol/seed mismatch");
    actor.update = in.value<int>();
    if (actor.update < 0 || in.value<uint32_t>() != 6 || in.value<uint32_t>() != 3)
        throw std::runtime_error("Wrong PPO model/action mapping");
    const auto task_version = in.value<uint32_t>();
    actor.reward_version = in.value<uint32_t>();
    if (task_version != 6 || (actor.reward_version != 4 && actor.reward_version != 5) ||
        in.value<uint32_t>() != 40 || in.value<uint32_t>() != 4)
        throw std::runtime_error("Wrong PPO task/reward/shape");
    if (schema >= 3) {
        if (in.value<uint32_t>() != 1)
            throw std::runtime_error("Wrong stochastic policy schema");
        actor.correlation = in.value<float>();
        if (!(actor.correlation == 0 || actor.correlation == .8f))
            throw std::runtime_error("Invalid stochastic correlation");
    }
    if (schema >= 4)
        actor.observation_schema = in.value<uint32_t>();
    if (!motor_observation::valid(actor.observation_schema) ||
        (expected_observations && actor.observation_schema != *expected_observations))
        throw std::runtime_error("PPO observation schema mismatch");
    auto finite = [](const std::vector<float> &values) {
        for (float value : values)
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite PPO checkpoint data");
    };
    actor.parameters = in.array<float>(2500);
    const auto critic = in.array<float>(2401), sigma = in.array<float>(4);
    if (actor.parameters.size() != 2500 || critic.size() != 2401 || sigma.size() != 4)
        throw std::runtime_error("Wrong PPO network dimensions");
    finite(actor.parameters);
    finite(critic);
    finite(sigma);
    actor.log_sigma = sigma;
    state.critic = critic;
    for (int index = 0; index < 2; ++index) {
        static_assert(sizeof(nn::AdamConfig) == 20, "PPO schema2 Adam layout changed");
        const auto config = in.value<nn::AdamConfig>();
        const int steps = in.value<int>();
        if (!std::isfinite(config.lr) || config.lr <= 0 || !std::isfinite(config.beta1) || config.beta1 < 0 ||
            config.beta1 >= 1 || !std::isfinite(config.beta2) || config.beta2 < 0 || config.beta2 >= 1 ||
            !std::isfinite(config.eps) || config.eps <= 0 || !std::isfinite(config.max_grad_norm) ||
            config.max_grad_norm < 0 || steps < 0)
            throw std::runtime_error("Invalid PPO optimizer metadata");
        const size_t count = index == 0 ? 2500 : 2401;
        const auto first = in.array<float>(count), second = in.array<float>(count);
        if (first.size() != count || second.size() != count)
            throw std::runtime_error("Wrong PPO optimizer dimensions");
        finite(first);
        finite(second);
        for (float value : second)
            if (value < 0)
                throw std::runtime_error("Invalid PPO second moments");
        (index == 0 ? actor.actor_steps : actor.critic_steps) = steps;
        auto &optimizer = index == 0 ? state.actor_adam : state.critic_adam;
        optimizer.config = config;
        optimizer.steps = steps;
        optimizer.first = first;
        optimizer.second = second;
    }
    in.finish();
    return state;
}
inline FrozenPpoActor read_ppo_actor(const std::vector<uint8_t> &payload, uint64_t expected_protocol,
                                     uint32_t expected_seed,
                                     uint32_t expected_observations = motor_observation::legacy) {
    return read_ppo_training_state(payload, expected_protocol, expected_seed, expected_observations).actor;
}
inline std::vector<uint8_t> write_ppo_training_state(const PpoTrainingState &state) {
    checkpoint::Writer out;
    out.value(uint32_t{0x31525050});
    out.value(uint32_t{4});
    out.value(state.actor.seed);
    out.value(state.source_protocol);
    out.value(state.actor.update);
    for (uint32_t v : {6u, 3u, 6u, state.actor.reward_version, 40u, 4u, 1u})
        out.value(v);
    out.value(state.actor.correlation);
    out.value(state.actor.observation_schema);
    out.array(state.actor.parameters);
    out.array(state.critic);
    out.array(state.actor.log_sigma);
    for (const auto *optimizer : {&state.actor_adam, &state.critic_adam}) {
        const auto bytes = adam_payload(*optimizer);
        out.data.insert(out.data.end(), bytes.begin(), bytes.end());
    }
    (void)read_ppo_training_state(out.data, state.source_protocol, state.actor.seed,
                                  state.actor.observation_schema);
    return out.data;
}
inline PpoTrainingState rescale_ppo_path_inputs(const PpoTrainingState &source) {
    if (source.actor.observation_schema != motor_observation::legacy)
        throw std::runtime_error("Path-input conversion requires legacy observation schema");
    auto converted = source;
    constexpr float factor = 36.f;
    auto change = [&](std::vector<float> &weights, PpoAdamState &adam) {
        for (int row = 0; row < 32; ++row)
            for (int col = 0; col < 3; ++col) {
                const int i = row * 40 + col;
                weights.at(i) /= factor;
                // Cotangent mapping: x'=s*x, W'=W/s => g'=s*g, m'=s*m, v'=s^2*v.
                // This preserves initial functions, NOT future scalar-LR/clipped Adam updates.
                adam.first.at(i) *= factor;
                adam.second.at(i) *= factor * factor;
            }
    };
    change(converted.actor.parameters, converted.actor_adam);
    change(converted.critic, converted.critic_adam);
    converted.actor.observation_schema = motor_observation::task_scale;
    return converted;
}
} // namespace cudarl::rl
#endif
