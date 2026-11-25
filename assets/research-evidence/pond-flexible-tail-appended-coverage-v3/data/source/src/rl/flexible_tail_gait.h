#pragma once

#include "rl/flexible_tail_launch.h"

namespace cudarl::flexible_tail {

inline constexpr float gait_phase_floor_rad = .08f;
inline constexpr float gait_nominal_rate = 2.f * pi;
inline constexpr float gait_seed_hidden_gain = .5f;

CUDARL_HD inline void observe_gait(const NavigationWorld& world, float (&out)[navigation_inputs]) {
    observe_launch(world, out);
    const auto& body = world.motion.body;
    const float yaw = body.angle[0];
    const float travel = body.velocity[6] / gait_nominal_rate;
    const float radius = sqrtf(yaw * yaw + travel * travel);
    const float scale = fmaxf(radius, gait_phase_floor_rad);
    out[motor_inputs] = yaw / scale;
    out[motor_inputs + 1] = travel / scale;
    out[motor_inputs + 2] = fmaxf(0.f, 1.f - radius / gait_phase_floor_rad);
}

} // namespace cudarl::flexible_tail
