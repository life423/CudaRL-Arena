#pragma once

#include "rl/pond_ppo_schedule.h"

namespace cudarl::rl::residual_turn_range {
namespace original = pond_ppo_schedule;
namespace pt = pond_ppo_task;
namespace p = physics;

// Adjacent worlds receive opposite signs at the same angle. Spread the first
// requests over all four angles so a short PPO block practices 60 and 90 degrees.
CUDARL_HD inline float requested_angle(uint32_t env, uint64_t issued) {
    const auto index = (env / 2u + issued) % 4u;
    const float degrees = index == 0 ? 15.f : index == 1 ? 30.f : index == 2 ? 60.f : 90.f;
    return ((env ^ issued) & 1u ? degrees : -degrees) * pt::obs::pi / 180.f;
}

CUDARL_HD inline void before(original::State& state, pt::Task& task,
                             uint32_t env, uint32_t seed, uint32_t program) {
    const bool issuing = task.kind == pt::Turn && !state.active &&
        task.episode_controls >= pt::turn_control && task.episode_controls < pt::episode_limit;
    if (!issuing) {
        original::before(state, task, env, seed, program);
        return;
    }
    const float angle = requested_angle(env, state.onsets);
    const bool first = state.serial == 0;
    const auto from = state.valid_course ? p::V{state.last_course_x,0,state.last_course_z}
                                         : task.request.request.heading;
    if (first) task.turn_radians = angle;
    original::before(state, task, env, seed, program);
    if (first) return;

    const auto heading = p::axis_rotation({0,1,0}, angle).rotate(from);
    task.request.request.heading = heading;
    task.request.request.terminal_velocity = heading * pt::target_speed_m_s;
    state.angle = task.turn_radians = angle;
}
}
