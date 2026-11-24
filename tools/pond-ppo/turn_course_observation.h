#pragma once

#include "rl/pond_ppo_schedule.h"

namespace cudarl::rl::pond_ppo_course_context {

// Actor-only adaptation of an already populated observation. The caller
// computes signed turn intent from the original heading in channel 41.
CUDARL_HD inline void apply(const pond_ppo_task::Task& task, const physics::BodyState& body,
                            pond_ppo_schedule::HorizontalCourse movement, float* actor_row) {
    if (task.kind != pond_ppo_task::Turn || !task.request_changed) return;

    const physics::V horizontal_motion{movement.x, 0, movement.z};
    const float speed = physics::norm(horizontal_motion);
    if (!physics::finite_number(speed) || speed < .003f) return;

    const physics::V course = horizontal_motion / speed;
    const auto velocity = task.request.request.terminal_velocity;
    const float requested_horizontal_speed = physics::norm({velocity.x, 0, velocity.z});
    const physics::V gait_velocity{course.x * requested_horizontal_speed,
                                   velocity.y, course.z * requested_horizontal_speed};
    pond_motor_observation::vector_features(actor_row, 3,
        body.orientation.inverse(gait_velocity) / pond_motor_observation::velocity_scale_m_s);
    pond_motor_observation::vector_features(actor_row, 6, body.orientation.inverse(course));
}

} // namespace cudarl::rl::pond_ppo_course_context
