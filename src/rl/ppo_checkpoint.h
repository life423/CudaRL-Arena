#ifndef CUDARL_RL_PPO_CHECKPOINT_H
#define CUDARL_RL_PPO_CHECKPOINT_H
#include "core/checkpoint.h"
#include "nn/adam_config.h"
#include <cmath>
namespace cudarl::rl {
struct FrozenPpoActor {
    uint32_t seed = 0;
    int update = 0, actor_steps = 0, critic_steps = 0;
    std::vector<float> parameters;
};
// PPO research schema1 fixes topology42-32-32-(2actor/1critic), float32,
// model6 and signed-cadence mapping2. The shared outer reader checks integrity.
// Parse/validate auxiliary state without constructing or restoring an optimizer.
inline FrozenPpoActor read_ppo_actor(const std::vector<uint8_t> &payload, uint64_t expected_protocol,
                                     uint32_t expected_seed) {
    checkpoint::Reader in{payload};
    if (in.value<uint32_t>() != 0x31525050 || in.value<uint32_t>() != 1)
        throw std::runtime_error("Wrong PPO research schema");
    FrozenPpoActor actor;
    actor.seed = in.value<uint32_t>();
    if (actor.seed != expected_seed || in.value<uint64_t>() != expected_protocol)
        throw std::runtime_error("PPO source protocol/seed mismatch");
    actor.update = in.value<int>();
    if (actor.update < 0 || in.value<uint32_t>() != 6 || in.value<uint32_t>() != 2)
        throw std::runtime_error("Wrong PPO model/action mapping");
    auto finite = [](const std::vector<float> &values) {
        for (float value : values)
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite PPO checkpoint data");
    };
    actor.parameters = in.array<float>(2498);
    const auto critic = in.array<float>(2465), sigma = in.array<float>(2);
    if (actor.parameters.size() != 2498 || critic.size() != 2465 || sigma.size() != 2)
        throw std::runtime_error("Wrong PPO network dimensions");
    finite(actor.parameters);
    finite(critic);
    finite(sigma);
    for (int index = 0; index < 2; ++index) {
        static_assert(sizeof(nn::AdamConfig) == 20, "PPO schema1 Adam layout changed");
        const auto config = in.value<nn::AdamConfig>();
        const int steps = in.value<int>();
        if (!std::isfinite(config.lr) || config.lr <= 0 || !std::isfinite(config.beta1) || config.beta1 < 0 ||
            config.beta1 >= 1 || !std::isfinite(config.beta2) || config.beta2 < 0 || config.beta2 >= 1 ||
            !std::isfinite(config.eps) || config.eps <= 0 || !std::isfinite(config.max_grad_norm) ||
            config.max_grad_norm < 0 || steps < 0)
            throw std::runtime_error("Invalid PPO optimizer metadata");
        const size_t count = index == 0 ? 2498 : 2465;
        const auto first = in.array<float>(count), second = in.array<float>(count);
        if (first.size() != count || second.size() != count)
            throw std::runtime_error("Wrong PPO optimizer dimensions");
        finite(first);
        finite(second);
        for (float value : second)
            if (value < 0)
                throw std::runtime_error("Invalid PPO second moments");
        (index == 0 ? actor.actor_steps : actor.critic_steps) = steps;
    }
    in.finish();
    return actor;
}
} // namespace cudarl::rl
#endif
